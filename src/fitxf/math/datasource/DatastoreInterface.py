import warnings
import logging
import threading
import os
import builtins
from fitxf.math.utils.Env import Env
from fitxf.math.utils.Logging import Logging


class DbParams:

    @staticmethod
    def get_db_params_from_envvars(
            identifier,
            db_create_tbl_sql,
            db_table,
            verify_certs = False,
    ):
        return DbParams(
            identifier = identifier,
            db_type = os.environ["DB_TYPE"],
            db_host = os.environ["DB_HOST"],
            db_port = os.environ["DB_PORT"],
            db_scheme = os.environ["DB_SCHEME"],
            db_username = os.environ["DB_USERNAME"],
            db_password = os.environ["DB_PASSWORD"],
            db_database = os.environ["DB_DATABASE"],
            db_root_folder = os.environ["DB_ROOT_FOLDER"],
            db_create_table_sql = db_create_tbl_sql,
            db_table = db_table,
            db_verify_certs = verify_certs,
        )

    def __init__(
            self,
            # any name to indentify this set of DB params
            identifier,
            db_type = None,
            db_host = None,
            db_port = None,
            db_scheme = None,
            db_username = None,
            db_password = None,
            db_database = None,
            db_verify_certs = True,
            db_table = None,
            db_create_table_sql = None,
            # for certain types of DB types like csv
            db_root_folder = None,
            params_other = None,
            logger = None,
    ):
        self.identifier = identifier
        self.db_type = db_type if db_type is not None else os.environ["DB_TYPE"]
        self.db_host = db_host if db_host is not None else os.environ["DB_HOST"]
        self.db_port = db_port if db_port is not None else os.environ["DB_PORT"]
        self.db_scheme = db_scheme if db_scheme is not None else os.environ["DB_SCHEME"]
        self.db_username = db_username if db_username is not None else os.environ["DB_USERNAME"]
        self.db_password = db_password if db_password is not None else os.environ["DB_PASSWORD"]
        self.db_database = db_database if db_database is not None else os.environ["DB_DATABASE"]
        self.db_verify_certs = db_verify_certs
        self.db_table = db_table
        self.db_create_table_sql = db_create_table_sql
        self.db_root_folder = db_root_folder if db_root_folder is not None else os.environ["DB_ROOT_FOLDER"]
        self.params_other = {} if params_other is None else params_other
        self.logger = logger if logger is not None else logging.getLogger()

        passwd = str(self.db_password)
        hide_len = min(4, int(0.5*len(passwd)))
        mask_len = len(passwd) - hide_len
        mask_len = max(0, mask_len)
        self.db_password_hide = str('*' * mask_len) + str(passwd[-hide_len:])
        self.logger.info('Initialized DB Params as: ' + str(self.get_db_info()))
        return

    def get_db_info(self):
        return \
            'DB Params for "' + str(self.identifier) + '" as follows: ' \
            + 'DB Type "' + str(self.db_type) + '", host "' + str(self.db_host) \
            + '", port ' + str(self.db_port) \
            + ', scheme "' + str(self.db_scheme) \
            + '", username "' + str(self.db_username)\
            + '", password "' + str(self.db_password_hide) \
            + '", database "' + str(self.db_database) \
            + '", verify certs "' + str(self.db_verify_certs) \
            + '", index "' + str(self.db_table) \
            + '", create table sql "' + str(self.db_create_table_sql) \
            + ', folder "' + str(self.db_root_folder) \
            + '", params other: ' + str(self.params_other)

    def copy_db_params_with_new_index(
            self,
            identifier,
            new_table,
    ):
        return DbParams(
            identifier = identifier,
            db_type = self.db_type,
            db_host = self.db_host,
            db_port = self.db_port,
            db_scheme = self.db_scheme,
            db_username = self.db_username,
            db_password = self.db_password,
            db_database = self.db_database,
            db_verify_certs = self.db_verify_certs,
            db_table = new_table,
            db_create_table_sql = self.db_create_table_sql,
            # for certain types of DB types like csv
            db_root_folder = self.db_root_folder,
            params_other = self.params_other,
            logger = self.logger,
        )


class DatastoreInterface:

    def __init__(
            self,
            # connection options, can pass during connection or here
            db_params: DbParams,
            ignore_warnings = False,
            logger = None,
    ):
        if ignore_warnings:
            # Ignore warning "InsecureRequestWarning: Unverified HTTPS request is being made to host 'localhost'..."
            warnings.filterwarnings("ignore")
        self.db_params = db_params
        self.logger = logger if logger is not None else logging.getLogger()

        self.con = None
        self.timeout = int(os.environ.get('DB_DEFAULT_TIMEOUT', 30))
        self.mutex_write = threading.Lock()
        return

    def connect(
            self,
    ):
        raise Exception('Must be implemented by derived class')

    def close(
            self,
    ):
        raise Exception('Must be implemented by derived class')

    def get(
            self,
            # e.g. {"answer": "take_seat"}
            match_phrase,
            match_condition: dict = {'and': True, 'exact': True},
            tablename = None,
            request_timeout = 20.0,
    ):
        raise Exception('Must be implemented by derived class')

    def get_all(
            self,
            key = None,
            max_records = 10000,
            tablename = None,
            request_timeout = 20.0,
    ):
        raise Exception('Must be implemented by derived class')

    def get_indexes(self):
        raise Exception('Must be implemented by derived class')

    def delete_index(
            self,
            tablename,
    ):
        raise Exception('Must be implemented by derived class')

    def get_column_names(
            self,
            tablename,
    ):
        records = self.get_all(
            tablename = tablename,
            max_records = 1,
        )
        assert len(records) > 0, 'Cannot get column names by default method without data for "' + str(tablename) + '"'
        return [k for k in records[0].keys()]

    def get_mapping(
            self,
            tablename = None,
    ):
        raise Exception('Must be implemented by derived class')

    def atomic_delete_add(
            self,
            delete_key: str,
            # list of dicts, python3.8 & below does not support this syntax
            records: list[dict],
            tablename: str = None,
    ):
        raise Exception('Must be implemented by derived class')

    def add(
            self,
            # list of dicts, python3.8 & below does not support this syntax
            records: list[dict],
            tablename: str = None,
    ):
        raise Exception('Must be implemented by derived class')

    def delete(
            self,
            match_phrase: dict,
            match_condition: dict = {'and': True, 'exact': True},
            tablename: str = None,
    ):
        raise Exception('Must be implemented by derived class')

    def delete_by_raw_query(
            self,
            raw_query,
            tablename = None,
    ):
        raise Exception('Must be implemented by derived class')

    def delete_all(
            self,
            tablename = None,
    ):
        raise Exception('Must be implemented by derived class')

    def add_column(
            self,
            colnew,
            data_type = str,
            tablename = None,
            default_value = None,
    ):
        raise Exception('Must be implemented by derived class')

    #
    # Set callback to receive data
    #
    def set_receive_callback(
            self,
            callback,
    ):
        raise Exception('Must be implemented by derived class')


class DatastoreInterfaceUnitTest:

    def __init__(
            self,
            ChildClass: type(DatastoreInterface),
            logger = None,
    ):
        self.ChildClass = ChildClass
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def get_child_class(
            self,
            tablename,
    ) -> DatastoreInterface:
        return self.ChildClass(
            db_params = DbParams.get_db_params_from_envvars(
                identifier = 'testdbif',
                db_create_tbl_sql = '',
                db_table = tablename,
            ),
            logger=self.logger,
        )

    def test(
            self,
            tablename,
    ):
        ds = self.get_child_class(tablename = tablename)
        ds.delete_index(tablename=tablename)
        ds.connect()

        rows = ds.get_all()
        assert len(rows) == 0, 'Should be empty but is not: ' + str(rows)

        new_records = [
            {'id': 100, 'text': "sentence A", 'answer': ' xx'},
            {'id': 101, 'text': "sentence A", 'answer': ' xx'},
            {'id': 102, 'text': "sentence B", 'answer': ' yy'},
            {'id': 103, 'text': "sentence B", 'answer': ' yy'},
            {'id': 103, 'text': "sentENce B", 'answer': ' zz'},
            {'id': 104, 'text': "this is sentence B", 'answer': ' zz'},
        ]
        ds.add(records=new_records)

        colnames = ds.get_column_names(tablename=tablename)
        self.logger.info('Column names derived: ' + str(colnames))
        assert colnames == ['id', 'text', 'answer'], 'Column names ' + str(colnames) + ' not expected'

        rows = ds.get_all()
        assert len(rows) == len(new_records), \
            'Length of memory cache ' + str(len(rows)) + ' not same with data added ' + str(len(new_records))
        for i, r in enumerate(rows):
            is_same_address = builtins.id(r) == builtins.id(new_records[i])
            test_ok = (not is_same_address)
            test_desc = 'Test for NOT SAME.'
            assert test_ok, \
                test_desc + ' Record #' + str(i) + ' same address = ' + str(is_same_address) + ': ' \
                + str(builtins.id(r)) + ' and ' + str(builtins.id(new_records[i])) + ' for record ' + str(r)

        for i, (cond, query, len_exp_recs, exp_remaining_ids, exp_remaining_texts) in enumerate([
            ({'and': True, 'exact': True}, {'id': 101}, 1, [100, 102, 103, 103, 104], None),
            ({'and': True, 'exact': True}, {'text': 'sentence B'}, 2, [100, 103, 104],  ['sentence A', 'sentENce B', 'this is sentence B']),
            ({'and': False, 'exact': True}, {'id': 104, 'text': 'sentence A'}, 2, [103], ['sentENce B',]),
            ({'and': False, 'exact': False}, {'text': 'B'}, 1, [], []),
        ]):
            self.logger.info('Records remaining now ' + str(ds.get_all()))
            recs_to_be_deleted = ds.get(
                match_phrase = query,
                match_condition = cond,
            )
            self.logger.info(
                '#' + str(i) + ' Records to be deleted for query "' + str(query) + '": ' + str(recs_to_be_deleted)
            )
            assert len(recs_to_be_deleted) == len_exp_recs, \
                '#' + str(i) + ' For id "' + str(id) + '" got != expected ' + str(len_exp_recs) + ' record, but ' \
                + str(len(recs_to_be_deleted)) + ': ' + str(recs_to_be_deleted)
            ds.delete(
                match_phrase = query,
                match_condition = cond,
            )
            self.logger.info('#' + str(i) + ' After DELETE ' + str(query) + '..')
            rows = ds.get_all()
            remaining_ids = [r['id'] for r in rows]
            remaining_texts = [r['text'] for r in rows]
            [print(r) for r in rows]
            assert remaining_ids == exp_remaining_ids, \
                '#' + str(i) + ' Remaining ids ' + str(remaining_ids) + ' not expected ' + str(exp_remaining_ids)

            if exp_remaining_texts is not None:
                assert remaining_texts == exp_remaining_texts, \
                    '#' + str(i) + ' Remaining texts ' + str(remaining_texts) \
                    + ' not expected ' + str(exp_remaining_texts)

        print('ALL TESTS PASSED OK')


if __name__ == '__main__':
    er = Env()
    Env.set_env_vars_from_file(env_filepath= er.REPO_DIR + '/.env.fitxf.math.ut')
    Logging.get_logger_from_env_var()
    dbp = DbParams.get_db_params_from_envvars(
        identifier = 'demo',
        db_create_tbl_sql = '',
        db_table = 'testtable'
    )
    exit(0)
