import logging


class MetadataInterface:

    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

    def __init__(
            self,
            # name to identify which user/table/etc this metadata is referring to
            user_id,
            metadata_tbl_name = 'model_metadata',
            logger = None,
    ):
        self.user_id = user_id
        self.metadata_tbl_name = metadata_tbl_name
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def get_metadata_db_data_last_update(
            self,
    ):
        raise Exception('Must be implemented by child class')

    def get_metadata_model_last_update(
            self,
    ):
        raise Exception('Must be implemented by child class')

    # signify that model has been updated
    def update_metadata_model_updated(
            self,
            llm_path: str,
            model_save_json_string: str,
    ):
        raise Exception('Must be implemented by child class')

    def update_metadata_db_raw_data_updated(self):
        raise Exception('Must be implemented by child class')

    def cleanup(
            self,
    ):
        raise Exception('Must be implemented by child class')

