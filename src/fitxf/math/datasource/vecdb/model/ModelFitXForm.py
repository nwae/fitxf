import time
import logging
import os
import numpy as np
from fitxf.math.datasource.DatastoreInterface import DbParams
from fitxf import FitXformInterface
from fitxf.math.lang.encode.LangModelPt import LangModelPt
from fitxf.math.datasource.vecdb.model.ModelInterface import ModelInterface, ModelEncoderInterface
from fitxf.math.datasource.vecdb.metadata.MetadataInterface import MetadataInterface
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Env import Env
from fitxf.math.utils.EnvironRepo import EnvRepo


class ModelFitTransform(ModelInterface):

    def __init__(
            self,
            user_id: str,
            llm_model: ModelEncoderInterface,
            col_content: str,
            col_label_user: str,
            col_label_std: str,
            col_embedding: str,
            numpy_to_b64_for_db: bool,
            vec_db_metadata: MetadataInterface,
            fit_xform_model: FitXformInterface,
            cache_tensor_to_file: bool,
            file_temp_dir: str,
            # allowed values: "np", "torch"
            return_tensors: str,
            enable_bg_thread_for_training: bool = True,
            logger = None,
    ):
        super().__init__(
            user_id = user_id,
            llm_model = llm_model,
            col_content = col_content,
            col_label_user = col_label_user,
            col_label_std = col_label_std,
            col_embedding = col_embedding,
            numpy_to_b64_for_db = numpy_to_b64_for_db,
            vec_db_metadata = vec_db_metadata,
            fit_xform_model = fit_xform_model,
            cache_tensor_to_file = cache_tensor_to_file,
            file_temp_dir = file_temp_dir,
            return_tensors = return_tensors,
            enable_bg_thread_for_training = enable_bg_thread_for_training,
            logger = logger,
        )

        self.model_grid_helper = self.fit_xform_model

        # 1st time load data
        self.init_data_model(
            max_tries = 1,
            background = False,
        )

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
            if not self.is_need_to_sync_model_with_db():
                continue
            try:
                self.update_model()
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

        self.update_model()
        return

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
    ):
        if not force_update:
            if not self.is_need_sync_db():
                self.logger.warning('No longer required to update model...')
                return False

        self.logger.info('Model updating...')
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
            n_cluster = min(unique_labels * 3, len(text_labels_user))
            self.logger.info('Fitting to labels of user: ' + str(self.text_labels_standardized))
            if len(text_encoded) > 0:
                res = self.model_grid_helper.fine_tune(
                    X = text_encoded,
                    X_labels = text_labels_user,
                    X_full_records = self.data_records,
                    n_components = n_cluster,
                )
                self.logger.debug(
                    'Fit to n cluster = ' + str(n_cluster) + ' result ' + str(res) + ', ' + str(text_labels_user)
                )

                model_save_b64json_string = self.model_grid_helper.model_to_b64json(
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

            self.last_sync_time_with_underlying_db = self.vec_db_metadata.get_metadata_db_data_last_update()
            if self.last_sync_time_with_underlying_db is None:
                self.logger.warning('Last DB data update time from metadata returned None')
                self.last_sync_time_with_underlying_db = self.OLD_DATETIME
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
            top_k = 5,
            # Instead of just returning the user labels, return full record. Applicable to some models only
            return_full_record = False,
    ):
        if self.is_need_sync_db():
            self.logger.info(
                'Model "' + str(self.model_grid_helper.__class__) + '" need update before prediction: '
                + str(text_list_or_embeddings)
            )
            self.update_model()
        else:
            self.logger.info(
                'Model "' + str(self.model_grid_helper.__class__) + '" dont need update before prediction: '
                + str(text_list_or_embeddings)
            )

        txt_lm = self.convert_to_embeddings_if_necessary(
            text_list_or_embeddings = text_list_or_embeddings,
        )

        #
        # There are 2 possible approaches, after obtaining the PCA segment numbers & relevant reference vectors:
        #    1. Transform input vector to PCA transform, then compare with the reference PCA transforms
        #    2. Do not transform, use original vectors to compare. For now, we use this approach to skip the step
        #       of transforming the input vector.
        #

        pred_labels_std_or_full_records, pred_probs = self.model_grid_helper.predict(
            X = txt_lm,
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

        txt_encoding = self.calc_embedding(content_list = [r[self.col_content] for r in records])
        self.logger.info(
            'Text encoded using lm model "' + str(self.llm_model.get_model_name()) + '" with shape '
            + str(txt_encoding.shape if self.return_tensors == 'np' else txt_encoding.size())
        )

        required_mutexes = [self.mutex_name_underlying_db]
        try:
            self.lock_mutexes.acquire_mutexes(
                id = 'delete',
                mutexes = required_mutexes,
            )
            records_with_embedding_and_labelstd = self.update_label_maps_from_new_recs__(
                records = records,
                text_encoding_tensor = txt_encoding,
            )
            self.delete_records_from_underlying_db__(
                match_phrases = [{delete_key: r[delete_key]} for r in records_with_embedding_and_labelstd],
            )
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

        txt_encoding = self.calc_embedding(content_list = [r[self.col_content] for r in records])
        self.logger.info(
            'Text encoded using lm model "' + str(self.llm_model.get_model_name()) + '" with shape '
            + str(txt_encoding.shape if self.return_tensors == 'np' else txt_encoding.size())
        )

        required_mutexes = [self.mutex_name_underlying_db]
        try:
            self.lock_mutexes.acquire_mutexes(
                id = 'add',
                mutexes = required_mutexes,
            )
            records_with_embedding_and_labelstd = self.update_label_maps_from_new_recs__(
                records = records,
                text_encoding_tensor = txt_encoding,
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
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    er = EnvRepo(repo_dir=os.environ.get("REPO_DIR", None))
    Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env.fitxf.math.ut')
    user_id = 'test_modelfitxf'
    db_prms = DbParams.get_db_params_from_envvars(
        identifier = 'test_modelfitxf',
        db_create_tbl_sql = None,
        db_table = user_id,
    )
    ModelFitTransform(
        user_id = user_id,
        llm_model = LangModelPt(
            model_name = 'test_modelfitxf',
            cache_folder = er.MODELS_PRETRAINED_DIR,
            logger = lgr,
        ),
        col_content = 'text',
        col_label_user = 'label',
        col_label_std = '__label',
        col_embedding = 'embedding',
    )
    exit(0)
