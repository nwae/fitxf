import logging
import os
from fitxf.math.datasource.DatastoreInterface import DbParams, DatastoreInterface


class ModelDbInterface:

    def __init__(
            self,
            tablename: str,
            col_content: str,
            col_content_type: str,
            col_label_user: str,
            col_label_standardized: str,
            col_embedding: str,
            max_records: int,
            logger = None,
    ):
        self.tablename = tablename
        self.col_content = col_content
        self.col_content_type = col_content_type
        self.col_label_user = col_label_user
        self.col_label_standardized = col_label_standardized
        self.col_embedding = col_embedding
        self.max_records = max_records
        self.logger = logger if logger is not None else logging.getLogger()

        create_tbl_sql = self.get_create_table_db_cmd(db_type=os.environ["DB_TYPE"])
        self.db_params = DbParams.get_db_params_from_envvars(
            identifier = tablename,
            db_create_tbl_sql = create_tbl_sql,
            db_table = tablename,
            verify_certs = os.environ.get("VERIFY_CERTS", "1") in ('1', 'yes', 'true'),
        )

        return

    def get_create_table_db_cmd(
            self,
            db_type,
    ) -> str:
        raise Exception('물려받는 클래스에서 구현해야 함')

    def get_db_params(self) -> DbParams:
        return self.db_params

    def get_underlying_db(self) -> DatastoreInterface:
        raise Exception('물려받는 클래스에서 구현해야 함')

    def insert(
            self,
            records: list[dict],
            tablename: str,
    ):
        raise Exception('물려받는 클래스에서 구현해야 함')

    def delete(
            self,
            match_phrase: dict,
            tablename: str,
    ):
        raise Exception('물려받는 클래스에서 구현해야 함')

    def load_data(
            self,
            max_attempts = 1,
    ):
        raise Exception('물려받는 클래스에서 구현해야 함')

    def connect_to_underlying_db(
            self,
    ):
        raise Exception('물려받는 클래스에서 구현해야 함')

    def convert_csv_string_array_to_float_array(
            self,
            string_array,
            custom_chars_remove,
    ):
        raise Exception('물려받는 클래스에서 구현해야 함')


if __name__ == '__main__':
    exit(0)
