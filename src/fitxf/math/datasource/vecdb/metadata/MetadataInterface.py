import logging
from datetime import datetime


class MetadataInterface:

    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

    COL_METADATA_INDEX = 'metadata_index'           # Index that this metadata refers to
    COL_METADATA_IDENTIFIER = 'metadata_identifier' # What kind of metadata (e.g. model, embedding size)
    COL_METADATA_TIMESTAMP = 'metadata_timestamp'
    COL_METADATA_VALUE = 'content'

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

    def get_timestamp_float(self):
        tref = datetime(year=2024, month=1, day=1)
        tnow = datetime.now()
        diff = tnow - tref
        return diff.total_seconds()

    def get_metadata(
            self,
            identifier
    ):
        raise Exception('Must be implemented by child class')

    # signify that model has been updated
    def update_metadata_identifier_value(
            self,
            identifier: str,
            value: str,
    ):
        raise Exception('Must be implemented by child class')

    def cleanup(
            self,
    ):
        raise Exception('Must be implemented by child class')

