import logging
import os
import re
import torch
import threading
import json
import numpy as np
from datetime import datetime
from fitxf import TensorUtils, FitXformInterface
from fitxf.math.algo.encoding.Base64 import Base64
from fitxf.math.datasource.vecdb.model.ModelEncoderInterface import ModelEncoderInterface
from fitxf.math.datasource.vecdb.model.ModelDbInterface import ModelDbInterface
from fitxf.math.datasource.vecdb.metadata.MetadataInterface import MetadataInterface
from fitxf.math.utils.Lock import Lock


class ModelInterface:

    OLD_DATETIME = datetime(year=1999, month=9, day=9)

    TYPE_TEXT = 'text'
    TYPE_IMG = 'image'
    TYPE_VOICE = 'voice'
    TYPE_SOUND = 'sound'
    TYPE_VIDEO = 'video'

    def __init__(
            self,
            user_id: str,
            # e.g. {'text': modeltxt, 'image': modelimg, ...}
            llm_model: dict[str, ModelEncoderInterface],
            model_db_class: type(ModelDbInterface),
            model_metadata_class: type(MetadataInterface),
            col_content: str,
            col_content_type: str,
            col_label_user: str,
            col_label_std: str,
            col_embedding: str,
            # <=0 means no check
            feature_len: int,
            numpy_to_b64_for_db: bool,
            fit_xform_model: FitXformInterface,
            cache_tensor_to_file: bool,
            file_temp_dir: str,
            # turn off any fitting or transform if False
            in_plain_or_test_mode: bool,
            # allowed values: "np", "torch"
            return_tensors: str = 'np',
            enable_bg_thread_for_training: bool = False,
            logger = None,
    ):
        self.user_id = user_id
        self.llm_model = llm_model
        self.model_db_class = model_db_class
        self.model_metadata_class = model_metadata_class
        self.col_content = col_content
        self.col_content_type = col_content_type
        self.col_label_user = col_label_user
        self.col_label_standardized = col_label_std
        self.col_embedding = col_embedding
        self.feature_len = feature_len
        self.numpy_to_b64_for_db = numpy_to_b64_for_db
        self.fit_xform_model = fit_xform_model
        self.cache_tensor_to_file = cache_tensor_to_file
        self.file_temp_dir = file_temp_dir
        self.in_plain_or_test_mode = in_plain_or_test_mode
        self.return_tensors = return_tensors
        self.enable_bg_thread_for_training = enable_bg_thread_for_training
        self.logger = logger if logger is not None else logging.getLogger()

        self.logger.info(
            'Model in plain/test mode set to "' + str(self.in_plain_or_test_mode) + '"'
        )

        self.base64_encoder = Base64(logger=self.logger)
        self.tensor_utils = TensorUtils(logger=self.logger)
        self.llm_model_path = {k: m.get_model_path() for k, m in self.llm_model.items()}

        self.model_db = self.__get_model_db(ModelClass=self.model_db_class)
        self.logger.info(
            'DB params "' + str(self.user_id) +'": ' + str(self.model_db.get_db_params().get_db_info())
        )

        self.vec_db_metadata = self.__get_model_metadata(MetadataClass=self.model_metadata_class)

        # Lock only by threads in the same worker
        self.mutex_name_model = 'model'
        self.mutex_name_underlying_db = 'db'
        self.lock_mutexes = Lock(
            # User may also provide their own storage (in the usual case)
            mutex_names = [self.mutex_name_model, self.mutex_name_underlying_db],
            logger = self.logger,
        )

        assert os.path.isdir(self.file_temp_dir), 'Not a directory for tmp caching "' + str(self.file_temp_dir) + '"'
        # Don't use characters like "." because during cleanup cant find
        self.file_name_prefix_for_match_pattern = 'VecDbModelStandard_' + \
                                str(datetime.now().strftime("%Y%m%d_%H%M%S")) + '_' \
                                + str(np.random.rand())[2:]
        self.file_name_prefix = self.file_temp_dir + '/' + self.file_name_prefix_for_match_pattern

        self.last_sync_time_with_underlying_db = self.OLD_DATETIME
        self.logger.info(
            'LLM model path "' + str(self.llm_model_path)
            + '", encode numpy to base 64 in DB = ' + str(self.numpy_to_b64_for_db)
        )

        self.bg_thread = threading.Thread(target=self.run_bg_thread)
        self.bg_thread_sleep_secs = float(os.environ.get("VECDB_BG_SLEEP_SECS", "10."))
        self.clear_memory_secs_inactive = float(os.environ.get("VECDB_CLEAR_MEMORY_SECS_INACTIVE", "30."))
        self.signal_stop_bg_thread = False
        self.logger.info(
            'Set bg thread sleep to ' + str(self.bg_thread_sleep_secs)
            + 's, clear memory inactive time to ' + str(self.clear_memory_secs_inactive) + 's.'
        )
        return

    def __get_model_db(
            self,
            ModelClass: type(ModelDbInterface),
    ) -> ModelDbInterface:
        return ModelClass(
            tablename = self.user_id,
            col_content = self.col_content,
            col_content_type = self.col_content_type,
            col_label_user = self.col_label_user,
            col_label_standardized = self.col_label_standardized,
            col_embedding = self.col_embedding,
            max_records = 999999,
            logger = self.logger,
        )

    def __get_model_metadata(
            self,
            MetadataClass: type(MetadataInterface),
    ) -> MetadataInterface:
        return MetadataClass(
            user_id = self.user_id,
            logger = self.logger,
        )

    def check_model_consistency_with_prev(
            self,
    ):
        # Verify that model name has not changed from previous
        try:
            row = self.vec_db_metadata.get_metadata(identifier='llm')
            self.logger.info('Row from metadata for llm: ' + str(row))
            model_prev = row[MetadataInterface.COL_METADATA_VALUE]
        except Exception as ex:
            self.logger.error('Error getting metadata model: ' + str(ex))
            model_prev = None
            # raise Exception(ex)
        self.logger.info('Previous LLM model from metadata "' + str(model_prev) + '"')

        # csv can return float type "nan" instead of None
        if model_prev is not None:
            name_prev = self.get_model_name_from_path(model_path=model_prev)
            name_cur = self.get_model_name_from_path(model_path=self.llm_model_path)
            assert name_prev == name_cur, \
                'Previous model different "' + str(model_prev) + '" (type "' + str(type(model_prev)) + \
                '") from new "' + str(self.llm_model_path) + '" (type "' + str(type(self.llm_model_path)) \
                + '")'
            self.logger.info(
                'Model consistency check ok, cur/prev name same "' + str(name_cur)
                + '", full path "' + str(self.llm_model_path) + '"'
            )
        else:
            self.logger.warning('Model from metadata returned None')

    def cleanup(self):
        try:
            files_in_tmp_dir = os.listdir(path=self.file_temp_dir)
            self.logger.info('Files in temp dir "' + str(self.file_temp_dir) + '": ' + str(files_in_tmp_dir))
            for f in files_in_tmp_dir:
                if re.match(pattern=str(self.file_name_prefix_for_match_pattern)+".*", string=f):
                    self.logger.info(
                        'Cleanup deleting file "' + str(f) + '" using prefix match pattern "'
                        + str(self.file_name_prefix_for_match_pattern) + '"'
                    )
                    os.remove(path=f)
                else:
                    self.logger.debug(
                        'Cleanup ignoring file "' + str(f) + '" using prefix match pattern "'
                        + str(self.file_name_prefix_for_match_pattern) + '"'
                    )
        except Exception as ex:
            self.logger.error('Error cleaning up temp dir "' + str(self.file_temp_dir) + '": ' + str(ex))
        return

    def stop_threads(self):
        if not self.signal_stop_bg_thread:
            self.signal_stop_bg_thread = True
            self.bg_thread.join()
            self.cleanup()
            self.logger.info('All threads successfully stopped')
        return

    def run_bg_thread(
            self,
    ):
        raise Exception('Must be implemented by derived class')

    def extend_feature_len(
            self,
            x: np.ndarray,
    ):
        # no check whatsoever if set to <= 0
        if self.feature_len <= 0:
            return x

        start_dim = x.ndim
        assert start_dim in [1, 2]

        x_tmp = x if start_dim==2 else np.resize(a=x, new_shape=(1, len(x)))
        assert x_tmp.ndim == 2

        h, l = x_tmp.shape
        if l < self.feature_len:
            x_ret = np.append(x_tmp, np.zeros(shape=(h, self.feature_len - l)), axis=-1)
        elif l > self.feature_len:
            raise Exception('Length of array ' + str(l) + ' longer than preset feature length ' + str(self.feature_len))
        else:
            x_ret = x_tmp
        x_final = x_ret if start_dim==2 else x_ret[0]
        # self.logger.debug('Returning extended vector of shape ' + str(x_final.shape))
        return x_final

    def calc_embedding(
            self,
            content_list,
            content_type,
    ):
        assert self.llm_model is not None, 'LLM model is None'
        content_encode = self.llm_model[content_type].encode(
            content_list = content_list,
            return_tensors = self.return_tensors,
        )
        content_encode_std_size = self.extend_feature_len(x=content_encode)
        embed_len = content_encode_std_size.shape[-1]
        if self.feature_len > 0:
            assert embed_len <= self.feature_len, \
                'Embedding length ' + str(embed_len) + ' cannot exceed given fixed feature len ' + str(self.feature_len)
        self.logger.info(
            'Calculated embeddings for content type "' + str(content_type)
            + '", size ' + str(content_encode_std_size.shape)
        )
        return content_encode_std_size

    def get_model_name_from_path(self, model_path):
        parts = [name for name in str(model_path).strip().split(sep=os.sep) if len(name) > 0]
        self.logger.info('Model path separated as ' + str(parts))
        return parts[-1]

    def update_metadata_model_updated(
            self,
            model_save_b64json_string = None,
    ):
        for id, val in (('model', model_save_b64json_string), ('llm', self.llm_model_path)):
            self.vec_db_metadata.update_metadata_identifier_value(
                identifier = id,
                value = val,
            )
        return

    def update_metadata_db_data_updated(self):
        self.vec_db_metadata.update_metadata_identifier_value(
            identifier = 'lastUpdateTimeDb',
            value = datetime.now().strftime(MetadataInterface.DATETIME_FORMAT),
        )
        return

    def init_data_model(
            self,
            max_tries = 1,
            background = False,
    ):
        raise Exception('Must be implemented by derived class')

    def load_data_model(
            self,
            max_tries = 1,
            background = False,
    ):
        raise Exception('Must be implemented by derived class')

    def reset_data_model__(self):
        self.map_lbl_to_idx = {}
        self.map_idx_to_lbl = {}
        self.text_labels_standardized = np.array([])
        self.data_records = []
        return

    def update_model(
            self,
            force_update = False,
            test_mode = False,
    ):
        raise Exception('Must be implemented by derived class')

    def set_high_urgency_to_sync_with_underlying_db(self):
        self.last_sync_time_with_underlying_db = self.OLD_DATETIME

    def get_data_length(self):
        return len(self.text_labels_standardized)

    def atomic_delete_add(
            self,
            delete_key: str,
            # list of dicts
            records: list[dict],
    ):
        raise Exception('Must be implemented by derived class')

    def add(
            self,
            # list of dicts
            records: list,
    ):
        raise Exception('Must be implemented by derived class')

    def delete(
            self,
            match_phrases,
    ):
        raise Exception('Must be implemented by derived class')

    def predict(
            self,
            text_list_or_embeddings,
            content_type: str = TYPE_TEXT,
            # can be cluster numbers to zoom into
            X_embeddings_local_space: np.ndarray = None,
            labels_local_space: list = None,
            top_k = 5,
            # Instead of just returning the user labels, return full record. Applicable to some models only
            return_full_record = False,
    ):
        raise Exception('Must be implemented by derived class')

    def is_updating_model(self):
        return self.lock_mutexes.is_locked(mutex=self.mutex_name_model)

    ############################################################################################################
    # Underlying DB
    ############################################################################################################

    def sync__db(
            self,
            max_tries = 1,
            background = False,
    ):
        raise Exception('Must be implemented by derived class')

    # Heuristic measure from 0 to 1, based on secs since last load time (proportional),
    # training data size (inversely proportional), etc
    def is_need_sync_db(
            self,
    ):
        # secs_since_last_load = Profiling().get_time_dif_secs(
        #     start = self.last_sync_time_with_underlying_db,
        #     stop = last_updated_time_underlying_db
        # )
        row = self.vec_db_metadata.get_metadata(
            identifier = 'lastUpdateTimeDb',
        )
        if row is None:
            db_last_update_time = None
            need_update = True
        else:
            db_last_update_time = datetime.strptime(
                row[MetadataInterface.COL_METADATA_VALUE], MetadataInterface.DATETIME_FORMAT
            )
            need_update = db_last_update_time > self.last_sync_time_with_underlying_db
        self.logger.info(
            'Need update = ' + str(need_update) + '. Last DB update time "' + str(db_last_update_time)
            + '", last sync time ' + str(self.last_sync_time_with_underlying_db)
        )

        if need_update:
            self.logger.info(
                'Need update. Last DB update time "' + str(db_last_update_time)
                + '", last sync time ' + str(self.last_sync_time_with_underlying_db)
            )
        return need_update

    def is_loading_model_from_underlying_db(self):
        return self.lock_mutexes.is_locked(mutex=self.mutex_name_underlying_db)

    def get_text_encoding_from_db_records(
            self,
            db_records,
    ) -> np.ndarray:
        # at the same time, remove embedding record from db_records
        text_encoded = [r.pop(self.col_embedding) for r in db_records]
        if self.numpy_to_b64_for_db:
            text_encoded = [
                # 3 steps: Encode from base 64 string to base 64 bytes, then decode the base 64 to actual bytes,
                # then finally convert to numpy from base 64 bytes
                self.base64_encoder.decode_base64_string_to_numpy_array(s64=s64, data_type='float64')
                for s64 in text_encoded
            ]
        if len(text_encoded) > 0:
            array_lengths = [len(v) for v in text_encoded]
            self.logger.info('Encoding lengths ' + str(array_lengths))
            max_l, min_l = np.max(array_lengths), np.min(array_lengths)
            if self.feature_len > 0:
                assert self.feature_len >= max_l
                if (self.feature_len != max_l) or (max_l != min_l):
                    text_encoded = [self.extend_feature_len(x=v) for v in text_encoded]
                    self.logger.warning(
                        'Appended all vectors to be same length ' + str(max_l) + ', array lengths now '
                        + str([len(v) for v in text_encoded])
                    )
        return np.array(text_encoded)

    def convert_to_embeddings_if_necessary(
            self,
            text_list_or_embeddings,
            content_type,
    ):
        txt_lm = None
        if type(text_list_or_embeddings) is np.ndarray:
            if text_list_or_embeddings.dtype in [
                'float64', 'float32', 'float', 'int64', 'int32', 'int',
                torch.float64, torch.float32, torch.float, torch.int64, torch.int32, torch.int,
            ]:
                txt_lm = text_list_or_embeddings
            else:
                self.logger.info(
                    'Passed in data type "' + str(text_list_or_embeddings.dtype) + '", assuming is text type'
                )

        if txt_lm is None:
            self.logger.info(
                'Passed in data is likely not embedding but text list, calculating embedding: '
                + str(text_list_or_embeddings)
            )
            txt_lm = self.calc_embedding(
                content_list = text_list_or_embeddings,
                content_type = content_type,
            )
        return txt_lm

    def __update_label_mapping(
            self,
            # when new data added, we might need to create new label mapping
            labels_from_new_records,
    ):
        for lb in labels_from_new_records:
            if lb not in self.map_lbl_to_idx:
                while True:
                    int_tmp = np.random.randint(low=0, high=2**31)
                    if int_tmp in self.map_idx_to_lbl.keys():
                        continue
                    self.map_lbl_to_idx[lb] = int_tmp
                    break
        self.map_idx_to_lbl = {i: lbl for lbl, i in self.map_lbl_to_idx.items()}
        return

    def update_label_maps_from_new_recs__(
            self,
            records,
            text_encoding_tensor,
    ):
        labels = [row[self.col_label_user] for row in records]
        self.__update_label_mapping(labels_from_new_records=labels)

        # Save records in append mode
        records_with_embedding = []
        for rec, txt_enc in zip(records, text_encoding_tensor):
            lbl_user = rec[self.col_label_user]
            # add standardized label to existing object in passed in records
            rec[self.col_label_standardized] = self.map_lbl_to_idx.get(lbl_user, -1)

            # Create a new record to be appended, don't add embedding to passed in records
            rec_copy = {k:v for k,v in rec.items()}
            rec_copy[self.col_embedding] = txt_enc
            records_with_embedding.append(rec_copy)

        return records_with_embedding

    def add_records_to_underlying_db__(
            self,
            records_with_embedding_and_labelstd,
    ):
        if self.numpy_to_b64_for_db:
            recs_for_storage_base64 = []
            for rec in records_with_embedding_and_labelstd:
                r = rec.copy()
                r[self.col_embedding] = self.base64_encoder.encode_numpy_array_to_base64_string(
                    x = r[self.col_embedding],
                    data_type = 'float64',
                )
                recs_for_storage_base64.append(r)
            self.logger.info(
                'Successfully encoded all embedding to base 64 string for storage.'
            )
        else:
            recs_for_storage_base64 = records_with_embedding_and_labelstd

        self.model_db.insert(
            records = recs_for_storage_base64,
            tablename = self.user_id,
        )
        self.logger.info(
            'Train update underlying DB saved ' + str(len(records_with_embedding_and_labelstd)) + ' records: '
            + str([{k: v for k, v in d.items() if k != self.col_embedding} for d in recs_for_storage_base64])
        )
        return

    def delete_records_from_underlying_db__(
            self,
            match_phrases,
    ):
        total_deleted = 0
        # Get all encoded texts, standardized labels, text encoding from memory cache
        for mp in match_phrases:
            self.__delete_single_record_from_underlying_db(
                match_phrase = mp,
            )
        return total_deleted

    def __delete_single_record_from_underlying_db(
            self,
            match_phrase,
    ):
        total_deleted = 0
        assert len(match_phrase) == 1, 'Not supported delete with more than 1 key phrase ' + str(match_phrase)
        self.logger.info('Trying to delete records using match phrase: ' + str(match_phrase))
        res_del = self.model_db.delete(
            match_phrase = match_phrase,
            tablename = self.user_id,
        )
        self.logger.info('Deleted record with match phrase ' + str(match_phrase) + ', result ' + str(res_del))
        return 1


if __name__ == '__main__':
    exit(0)
