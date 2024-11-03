import time
import logging
import os
import numpy as np
from datetime import datetime
from fitxf.math.datasource.DatastoreInterface import DbParams
from fitxf import FitXformInterface
from fitxf.math.lang.encode.LangModelPt import LangModelPt
from fitxf.math.lang.encode.ImgPt import ImgPt
from fitxf.math.datasource.vecdb.model.ModelDbInterface import ModelDbInterface
from fitxf.math.datasource.vecdb.model.ModelInterface import ModelInterface, ModelEncoderInterface
from fitxf.math.datasource.vecdb.metadata.MetadataInterface import MetadataInterface
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Env import Env


class ModelFitTransform(ModelInterface):

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
        super().__init__(
            user_id = user_id,
            llm_model = llm_model,
            model_db_class = model_db_class,
            model_metadata_class = model_metadata_class,
            col_content = col_content,
            col_content_type = col_content_type,
            col_label_user = col_label_user,
            col_label_std = col_label_std,
            col_embedding = col_embedding,
            feature_len = feature_len,
            numpy_to_b64_for_db = numpy_to_b64_for_db,
            fit_xform_model = fit_xform_model,
            cache_tensor_to_file = cache_tensor_to_file,
            file_temp_dir = file_temp_dir,
            in_plain_or_test_mode = in_plain_or_test_mode,
            return_tensors = return_tensors,
            enable_bg_thread_for_training = enable_bg_thread_for_training,
            logger = logger,
        )

        # 1st time load data
        self.init_data_model(
            max_tries = 1,
            background = False,
        )
        self.check_model_consistency_with_prev()

        if self.enable_bg_thread_for_training:
            self.bg_thread.start()
        return

    def stop_threads(self):
        if not self.signal_stop_bg_thread:
            self.signal_stop_bg_thread = True
            if self.enable_bg_thread_for_training:
                self.bg_thread.join()
            self.logger.info('All threads successfully stopped')
        return

    def run_bg_thread(
            self,
    ):
        self.logger.info('Model thread started...')
        while True:
            time.sleep(self.bg_thread_sleep_secs)
            if self.signal_stop_bg_thread:
                self.logger.warning('Exiting model thread, stop signal received.')
                break
            if not self.is_need_sync_db():
                continue
            try:
                self.update_model(
                    test_mode = self.in_plain_or_test_mode,
                )
            except Exception as ex:
                self.logger.error('Error updating model compression from scheduled job: ' + str(ex))

    def init_data_model(
            self,
            max_tries = 1,
            background = False,
    ):
        # During initialization, we cannot throw exception. Fail means will depend on primary user cache already.
        required_mutexes = [self.mutex_name_model]
        try:
            self.lock_mutexes.acquire_mutexes(
                id = 'init_data_model',
                mutexes = required_mutexes,
            )
            self.__reset_data_model()
            # let Exceptions throw
        finally:
            self.lock_mutexes.release_mutexes(mutexes=required_mutexes)

        # get previous model data
        mtd_row = self.vec_db_metadata.get_metadata(identifier='model')
        self.logger.info(
            'Previous model type "' + str(type(mtd_row)) + '" info from metadata: '
            + str(mtd_row)
        )
        if not self.base64_encoder.is_base_64_string(mtd_row):
            self.logger.info(
                'Invalid model base64 string "' + str(mtd_row) + '", proceed to update model..'
            )
            self.update_model(
                test_mode = self.in_plain_or_test_mode,
            )
        else:
            try:
                self.lock_mutexes.acquire_mutexes(
                    id = 'init_data_model',
                    mutexes = required_mutexes,
                )
                model_b64json_str = mtd_row[MetadataInterface.COL_METADATA_VALUE]
                self.fit_xform_model.load_model_from_b64json(
                    model_b64json = model_b64json_str,
                )
                self.text_labels_standardized = np.array(self.fit_xform_model.X_labels)
                self.last_sync_time_with_underlying_db = self.vec_db_metadata.get_datetime_from_timestamp(
                    timestamp = mtd_row[MetadataInterface.COL_METADATA_TIMESTAMP]
                )
                self.logger.info(
                    'Loaded model ' + str(self.fit_xform_model.__class__)
                    + ' from saved model with timestamp ' + str(self.last_sync_time_with_underlying_db)
                    + ' metadata: ' + str(model_b64json_str)
                )
            finally:
                self.lock_mutexes.release_mutexes(mutexes=required_mutexes)
        return

    def load_data_model(
            self,
            max_tries = 1,
            background = False,
    ):
        return self.init_data_model(
            max_tries = max_tries,
            background = background,
        )

    def __reset_data_model(
            self,
    ):
        #
        # Default "built-in filesystem" model parameters required for inference
        #
        # In truth these maps are not required for this simple dense model, but we keep it anyway
        # since all other proper math models (e.g. NN) will always need mapping labels to numbers.
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
        if not force_update:
            if not self.is_need_sync_db():
                self.logger.warning('No longer required to update model...')
                return False

        self.logger.info('Model updating, test mode ' + str(test_mode) + '.')

        # Lock also underlying DB mutex because our metadata is also stored there
        required_mutexes = [self.mutex_name_model, self.mutex_name_underlying_db]
        try:
            self.lock_mutexes.acquire_mutexes(
                id = 'update_model',
                mutexes = required_mutexes,
            )
            db_records = self.model_db.load_data(max_attemps=1)

            self.data_records = [
                {
                    k: v for k, v in r.items()
                    if k in [self.col_content, self.col_label_standardized, self.col_label_user, self.col_embedding]
                }
                for r in db_records
            ]

            text_encoded = self.get_text_encoding_from_db_records(db_records=self.data_records)
            text_labels_user = [r[self.col_label_user] for r in self.data_records]
            self.text_labels_standardized = np.array([r[self.col_label_standardized] for r in self.data_records])

            unique_labels = len(np.unique(np.array(text_labels_user)))

            # not more than 3 per label, or not more than half the data length
            n_cluster = min(unique_labels * 3, int(len(text_labels_user)/2))
            self.logger.info(
                'Empirical estimation of total clusters ' + str(n_cluster) + ' from unique labels ' + str(unique_labels)
                + ', length of data ' + str(len(text_labels_user))
            )

            self.logger.info('Fitting to labels of user: ' + str(self.text_labels_standardized))
            if len(text_encoded) > 0:
                res = self.fit_xform_model.fine_tune(
                    X = text_encoded,
                    X_labels = text_labels_user,
                    X_full_records = self.data_records,
                    n_components = n_cluster,
                    test_mode = test_mode,
                )
                self.logger.debug(
                    'Fit to n cluster = ' + str(n_cluster) + ' result ' + str(res) + ', ' + str(text_labels_user)
                )

                model_save_b64json_string = self.fit_xform_model.model_to_b64json(
                    numpy_to_base64_str = True,
                    dump_to_b64json_str = True,
                )
                self.logger.debug(
                    'For user id "' + str(self.user_id) + '" model save json string: ' + str(model_save_b64json_string)
                )
                self.update_metadata_model_updated(
                    model_save_b64json_string = model_save_b64json_string,
                )
            else:
                # Update model metadata so that it will signal that we are done with loading data
                # Otherwise this function will be called non-stop
                self.update_metadata_model_updated(
                    model_save_b64json_string = None,
                )
                self.logger.info('No data or dont exist yet for "' + str(self.user_id) + '", nothing to fit.')

            mtd_row = self.vec_db_metadata.get_metadata(
                identifier = 'lastUpdateTimeDb',
            )
            if mtd_row is None:
                self.logger.warning('Last DB data update time from metadata returned None')
                self.last_sync_time_with_underlying_db = self.OLD_DATETIME
            else:
                self.last_sync_time_with_underlying_db = datetime.strptime(
                    mtd_row[MetadataInterface.COL_METADATA_VALUE], MetadataInterface.DATETIME_FORMAT
                )
            self.logger.info(
                'DB params: ' + str(self.model_db.get_db_params().get_db_info())
                + ', encode embedding to base 64 = ' + str(self.numpy_to_b64_for_db)
                + ', updated last sync time DB to "' + str(self.last_sync_time_with_underlying_db) + '"'
            )
            return True
        finally:
            self.lock_mutexes.release_mutexes(mutexes=required_mutexes)

    # Not necessarily faster, but will reduce RAM footprint
    def predict(
            self,
            text_list_or_embeddings,
            content_type: str = ModelInterface.TYPE_TEXT,
            # can be cluster numbers to zoom into
            X_embeddings_local_space: np.ndarray = None,
            labels_local_space: list = None,
            top_k = 5,
            # Instead of just returning the user labels, return full record. Applicable to some models only
            return_full_record = False,
    ):
        if self.is_need_sync_db():
            self.logger.info(
                'Model "' + str(self.fit_xform_model.__class__) + '" need update before prediction: '
                + str(text_list_or_embeddings)
            )
            self.update_model(
                test_mode = self.in_plain_or_test_mode,
            )
        else:
            self.logger.info(
                'Model "' + str(self.fit_xform_model.__class__) + '" dont need update before prediction: '
                + str(text_list_or_embeddings)
            )

        txt_lm = self.convert_to_embeddings_if_necessary(
            text_list_or_embeddings = text_list_or_embeddings,
            content_type = content_type,
        )
        # txt_lm = self.extend_feature_len(x=txt_lm)

        #
        # There are 2 possible approaches, after obtaining the PCA segment numbers & relevant reference vectors:
        #    1. Transform input vector to PCA transform, then compare with the reference PCA transforms
        #    2. Do not transform, use original vectors to compare. For now, we use this approach to skip the step
        #       of transforming the input vector.
        #

        pred_labels_std_or_full_records, pred_probs = self.fit_xform_model.predict(
            X = txt_lm,
            X_search_local_space = X_embeddings_local_space,
            labels_search_local_space = labels_local_space,
            top_k = top_k,
            return_full_record = return_full_record,
        )

        return pred_labels_std_or_full_records, pred_probs

    def atomic_delete_add(
            self,
            delete_key: str,
            # list of dicts
            records: list[dict],
    ):
        assert len(records) > 0, 'No records to train'
        self.logger.info('Add records of length ' + str(len(records)))

        cont_types = list(np.unique([r[self.col_content_type] for r in records]))
        if len(cont_types) > 1:
            encodings = []
            for i, row in enumerate(records):
                row_encode = self.calc_embedding(
                    content_list = [row[self.col_content]],
                    content_type = row[self.col_content_type],
                )
                # row_encode = self.extend_feature_len(x=row_encode)
                encodings.append(row_encode)
            content_encoding = np.vstack(encodings)
        else:
            content_encoding = self.calc_embedding(
                content_list = [r[self.col_content] for r in records],
                content_type = cont_types[0],
            )
            # content_encoding = self.extend_feature_len(x=content_encoding)
        self.logger.info(
            'Content of types ' + str(cont_types) + ' encoded using lm model "' + str(self.llm_model) + '" with shape '
            + str(content_encoding.shape if self.return_tensors == 'np' else content_encoding.size())
        )
        # txt_encoding = self.calc_embedding(
        #     content_list = [r[self.col_content] for r in records],
        # )
        self.logger.info(
            'Text encoded using lm model "' + str(self.llm_model) + '" with shape '
            + str(content_encoding.shape if self.return_tensors == 'np' else content_encoding.size())
        )

        required_mutexes = [self.mutex_name_underlying_db]
        try:
            self.lock_mutexes.acquire_mutexes(
                id = 'delete',
                mutexes = required_mutexes,
            )
            records_with_embedding_and_labelstd = self.update_label_maps_from_new_recs__(
                records = records,
                text_encoding_tensor = content_encoding,
            )
            mps = [{delete_key: r[delete_key]} for r in records_with_embedding_and_labelstd]
            try:
                self.delete_records_from_underlying_db__(
                    match_phrases = mps,
                )
            except Exception as ex_del:
                self.logger.info('Ignore delete error before add using match phrases ' + str(mps) + ': ' + str(ex_del))
                pass
            self.add_records_to_underlying_db__(
                records_with_embedding_and_labelstd = records_with_embedding_and_labelstd,
            )
            self.update_metadata_db_data_updated()
        finally:
            self.lock_mutexes.release_mutexes(mutexes=required_mutexes)

        # if not self.enable_bg_thread_for_training:
        #     self.update_model()
        #     is_updated = self.update_model()
        #     self.logger.info(
        #         'Model is updated = "' + str(is_updated) + '" after add records '
        #         + str([r[self.col_content] for r in records])
        #     )
        return

    def add(
            self,
            # list of dicts
            records: list,
    ):
        assert len(records) > 0, 'No records to train'
        self.logger.info('Add records of length ' + str(len(records)))
        cont_types = list(np.unique([r[self.col_content_type] for r in records]))
        if len(cont_types) > 1:
            encodings = []
            for i, row in enumerate(records):
                row_encode = self.calc_embedding(
                    content_list = [row[self.col_content]],
                    content_type = row[self.col_content_type],
                )
                # row_encode = self.extend_feature_len(x=row_encode)
                encodings.append(row_encode)
            content_encoding = np.vstack(encodings)
        else:
            content_encoding = self.calc_embedding(
                content_list = [r[self.col_content] for r in records],
                content_type = cont_types[0],
            )
            # content_encoding = self.extend_feature_len(x=content_encoding)
        self.logger.info(
            'Content of types ' + str(cont_types) + ' encoded using lm model "' + str(self.llm_model) + '" with shape '
            + str(content_encoding.shape if self.return_tensors == 'np' else content_encoding.size())
        )

        required_mutexes = [self.mutex_name_underlying_db]
        try:
            self.lock_mutexes.acquire_mutexes(
                id = 'add',
                mutexes = required_mutexes,
            )
            records_with_embedding_and_labelstd = self.update_label_maps_from_new_recs__(
                records = records,
                text_encoding_tensor = content_encoding,
            )
            self.add_records_to_underlying_db__(
                records_with_embedding_and_labelstd = records_with_embedding_and_labelstd,
            )
            self.update_metadata_db_data_updated()
        finally:
            self.lock_mutexes.release_mutexes(mutexes=required_mutexes)

        # if not self.enable_bg_thread_for_training:
        #     is_updated = self.update_model()
        #     self.logger.info(
        #         'Model is updated = "' + str(is_updated) + '" after add records '
        #         + str([r[self.col_content] for r in records])
        #     )
        return

    def delete(
            self,
            match_phrases,
    ):
        required_mutexes = [self.mutex_name_underlying_db]
        try:
            self.lock_mutexes.acquire_mutexes(
                id = 'delete',
                mutexes = required_mutexes,
            )
            self.delete_records_from_underlying_db__(
                match_phrases = match_phrases,
            )
            self.update_metadata_db_data_updated()
        finally:
            self.lock_mutexes.release_mutexes(mutexes=required_mutexes)

        # if not self.enable_bg_thread_for_training:
        #     self.update_model()
        #     is_updated = self.update_model()
        #     self.logger.info(
        #         'Model is updated = "' + str(is_updated) + '" after delete records ' + str(match_phrases)
        #     )
        return


if __name__ == '__main__':
    from fitxf import FitXformClusterCosine, FitXformCluster, FitXformPca
    from nwae.math.datasource.vecdb.model.ModelDb import ModelDb
    from nwae.math.datasource.vecdb.metadata.Metadata import ModelMetadata
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    er = Env()
    Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env.fitxf.math.ut')
    user_id = 'test_modelfitxf'
    db_prms = DbParams.get_db_params_from_envvars(
        identifier = 'test_modelfitxf',
        db_create_tbl_sql = None,
        db_table = user_id,
    )
    model_fit = ModelFitTransform(
        user_id = user_id,
        llm_model = {
            ModelInterface.TYPE_TEXT: LangModelPt(
                # model_name = 'intfloat/multilingual-e5-small',
                # model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                cache_folder = er.MODELS_PRETRAINED_DIR,
                logger = lgr,
            ),
            ModelInterface.TYPE_IMG: ImgPt(
                cache_folder = er.MODELS_PRETRAINED_DIR,
                logger = lgr,
            )
        },
        model_db_class = ModelDb,
        model_metadata_class = ModelMetadata,
        col_content = 'c',
        col_content_type = 't',
        col_label_user = 'l',
        col_label_std = '__l',
        col_embedding = 'e',
        feature_len = 384,
        fit_xform_model = FitXformCluster(logger=lgr),
        numpy_to_b64_for_db = True,
        file_temp_dir = er.REPO_DIR + '/tmp',
        cache_tensor_to_file = False,
        enable_bg_thread_for_training = False,
        logger = lgr,
    )
    recs_test = [
        {'l': 'fruits', 't': 'text', 'c': 'mango'}, {'l': 'fruits', 't': 'text', 'c': 'pineapple'},
        {'l': 'fruits',  't': 'text', 'c': 'apple'}, {'l': 'fruits',  't': 'text', 'c': 'orange'},
        {'l': 'beer',  't': 'text', 'c': 'Мюнхенский хеллес или Хелль '},
        {'l': 'beer', 't': 'text', 'c': 'Грузинский Черный Лев'},
        {'l': 'plov', 't': 'image',
         'c': 'https://img.freepik.com/premium-photo/shakh-plov-cooked-rice-dish-with-raisins-beautiful-plate-islamic-arabic-food_1279579-5074.jpg?w=1800'},
        {'l': 'plov', 't': 'image',
         'c': 'https://img.freepik.com/premium-psd/tasty-fried-vegetable-rice-plate-isolated-transparent-background_927015-3126.jpg?w=1480'},
    ]
    model_fit.atomic_delete_add(records=recs_test, delete_key='c')

    for test in [
        'давай выпем',
        'https://www.alyonascooking.com/wp-content/uploads/2019/05/plov-recipe-6.jpg',
    ]:
        res = model_fit.predict(text_list_or_embeddings=[test], top_k=2)
        print(res)
    exit(0)
