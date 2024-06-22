import logging
from datetime import datetime


class MetadataInterface:

    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

    COL_METADATA_USERID = 'metadata_userid'
    COL_METADATA_IDENTIFIER = 'metadata_identifier'
    COL_METADATA_TIMESTAMP = 'metadata_timestamp'
    COL_METADATA_VALUE = 'metadata_value'

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

    def get_timestamp(
            self,
            dt: datetime = None,
    ):
        dt = datetime.now() if dt is None else dt
        return datetime.timestamp(dt)

    def get_datetime_from_timestamp(
            self,
            timestamp: float,
    ):
        return datetime.fromtimestamp(timestamp)

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


if __name__ == '__main__':
    m = MetadataInterface(user_id='main')
    now = datetime.now()
    ts = m.get_timestamp(dt=now)
    dt = m.get_datetime_from_timestamp(timestamp=ts)
    print(ts, dt, now)
    exit(0)
