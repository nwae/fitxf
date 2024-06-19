import logging
import os
import re
import numpy as np
import pandas as pd
import threading
from fitxf.math.utils.LockF import LockFile
from fitxf.math.datasource.DatastoreInterface import DatastoreInterface, DbParams, DatastoreInterfaceUnitTest
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Env import Env
from fitxf.math.utils.EnvironRepo import EnvRepo


class Csv(DatastoreInterface):

    def __init__(
            self,
            db_params: DbParams,
            ignore_warnings = False,
            logger = None,
    ):
        super().__init__(
            db_params = db_params,
            ignore_warnings = ignore_warnings,
            logger = logger,
        )

        self.filepath = self.db_params.db_root_folder + os.sep + self.db_params.db_table

        self.use_file_lock = os.environ.get("CSV_USE_FILE_LOCK", "1").lower() in ("true", "1", "yes",)
        self.__lock_file_path = \
            db_params.db_root_folder + '/' + re.sub(pattern="[/\\\\:]", repl="_", string=self.db_params.db_table)
        self.__lock_file = LockFile(
            lock_file_path = self.__lock_file_path,
            logger = self.logger,
        )
        self.logger.info(
            'Using lock "' + str(self.use_file_lock) + '", file "' + str(self.__lock_file_path)
            + '" for file "' + str(self.filepath) + '"'
        )

        # This mutex will only work by thread/worker, for multithread/worker use file locks or RabbitMq
        self.__mutex = threading.Lock()
        return

    def __acquire_lock(self):
        if self.use_file_lock:
            self.__lock_file.acquire_file_cache_lock()
        self.__mutex.acquire()
        return

    def __release_lock(self):
        if self.use_file_lock:
            self.__lock_file.release_file_cache_lock()
        self.__mutex.release()
        return

    def connect(
            self,
    ):
        if not os.path.exists(self.filepath):
            self.logger.info(
                'Csv filepath does not exist "' + str(self.filepath) + '"'
            )
        return

    def __get_full_path(
            self,
            tablename,
    ):
        tablename = self.filepath if tablename is None else tablename

        if tablename != self.filepath:
            fullpath = self.db_params.db_root_folder + os.sep + str(tablename)
        else:
            fullpath = tablename
        assert fullpath == self.filepath
        return fullpath

    def get(
            self,
            # e.g. {"answer": "take_seat"}
            match_phrase,
            match_condition = 'AND',
            tablename = None,
            request_timeout = 20.0,
    ):
        tablename = self.__get_full_path(tablename=tablename)

        try:
            self.__acquire_lock()
            df = pd.read_csv(tablename, index_col=False)
            self.logger.debug('Dataframe read from "' + str(tablename) + '", shape ' + str(df.shape))
            condition = True if match_condition == 'AND' else False
            for c,v in match_phrase.items():
                if match_condition == 'AND':
                    condition = condition & (df[c] == v)
                else:
                    condition = condition | (df[c] == v)
                # self.logger.info('Condition ' + str(condition) + ' on ' + str(df))
            df_filter = df[condition]
            return df_filter.to_dict(orient='records')
        except Exception as ex:
            self.logger.error('Error reading data from table/index "' + str(tablename) + '": ' + str(ex))
            return []
        finally:
            self.__release_lock()

    def get_all(
            self,
            key = None,
            max_records = 10000,
            tablename = None,
            request_timeout = 20.0,
    ):
        tablename = self.__get_full_path(tablename=tablename)

        if not os.path.exists(tablename):
            self.logger.error('Read path do not exist "' + str(tablename) + '"')
            return []
        try:
            self.__acquire_lock()
            df = pd.read_csv(tablename, index_col=False)
            self.logger.info('Dataframe read from "' + str(tablename) + '", shape ' + str(df.shape))
            return df.to_dict(orient='records')
        except Exception as ex:
            self.logger.error('Error reading from csv filepath "' + str(tablename) + '": ' + str(ex))
            return []
        finally:
            self.__release_lock()

    def get_indexes(self):
        return 'Delete files in folder yourself, this function not supported'

    def delete_index(
            self,
            tablename,
    ):
        tablename = self.__get_full_path(tablename=tablename)
        try:
            os.remove(tablename)
            self.logger.info('Deleted CSV index "' + str(tablename) + '"')
        except Exception as ex:
            self.logger.error('Error deleting CSV index "' + str(tablename) + '": ' + str(ex))
        return True

    def atomic_delete_add(
            self,
            delete_key: str,
            # list of dicts
            records: list[dict],
            tablename: str = None,
    ):
        tablename = self.__get_full_path(tablename=tablename)

        try:
            self.__acquire_lock()
            for rec in records:
                self.__delete(
                    match_phrase = rec,
                    match_condition = 'AND',
                    tablename = tablename,
                )
                return self.__add(
                    records = [rec],
                    tablename = tablename,
                )
        except Exception as ex:
            self.logger.error('Error occurred: ' + str(ex))
            raise Exception(ex)
        finally:
            self.__release_lock()

    def add(
            self,
            # list of dicts
            records,
            tablename = None,
    ):
        tablename = self.__get_full_path(tablename=tablename)

        try:
            self.__acquire_lock()
            self.__add(
                records = records,
                tablename = tablename,
            )
        except Exception as ex:
            errmsg = \
                'Error occurred table "' + str(tablename) + '" add records ' + str(records) + ', exception: ' + str(ex)
            self.logger.error(errmsg)
            raise Exception(errmsg)
        finally:
            self.__release_lock()

    def __add(
            self,
            # list of dicts
            records: list[dict],
            tablename: str,
    ):
        df_new = pd.DataFrame(records)
        new_record_columns = df_new.columns.tolist()

        if not os.path.exists(tablename):
            df_csv = pd.DataFrame(columns=new_record_columns)
            csv_columns = new_record_columns
            self.logger.info('Using new records columns: ' + str(new_record_columns))
        else:
            # this line will throw error if no columns or file is empty
            df_csv = pd.read_csv(tablename, index_col=False)
            existing_columns = df_csv.columns.tolist()
            csv_columns = existing_columns if len(existing_columns) > 0 else new_record_columns
            self.logger.info('Using columns: ' + str(csv_columns))

        missing_columns_in_records = [col for col in csv_columns if col not in new_record_columns]
        if missing_columns_in_records:
            self.logger.warning('Missing columns in records to be added: ' + str(missing_columns_in_records))
            for col in missing_columns_in_records:
                df_new[col] = None
                self.logger.warning('Added new column "' + str(col) + '" to new records/dataframe to be added')

        # make sure the columns are in the same order
        df_new = df_new[df_csv.columns]
        df = pd.concat([df_csv, df_new])
        df.to_csv(tablename, index=False)
        self.logger.info('Written total records ' + str(len(df)) + ' to file "' + str(tablename) + '"')
        return

    def delete(
            self,
            match_phrase,
            match_condition = 'AND',
            tablename = None,
    ):
        tablename = self.__get_full_path(tablename=tablename)
        try:
            self.__acquire_lock()
            self.__delete(
                match_phrase = match_phrase,
                match_condition = match_condition,
                tablename = tablename,
            )
        except Exception as ex:
            errmsg = \
                'Error occurred table "' + str(tablename) + '" delete ' + str(match_phrase) + ', exception: ' + str(ex)
            self.logger.error(errmsg)
            raise Exception(errmsg)
        finally:
            self.__release_lock()

    def __delete(
            self,
            match_phrase: dict,
            match_condition: str,
            tablename: str,
    ):
        df = pd.read_csv(tablename, index_col=False)
        self.logger.debug('Dataframe read from "' + str(tablename) + '", shape ' + str(df.shape))
        condition = True if match_condition == 'AND' else False
        for c, v in match_phrase.items():
            if match_condition == 'AND':
                condition = condition & (df[c] == v)
            else:
                condition = condition | (df[c] == v)
            # self.logger.info('Condition ' + str(condition) + ' on ' + str(df))
        rows_removed = len(df[condition])
        df_remaining = df[np.logical_not(condition)]
        df_remaining.to_csv(tablename, index=False)
        return {'deleted': rows_removed}

    def add_column(
            self,
            colnew,
            data_type = str,
            tablename = None,
            default_value = None,
    ):
        tablename = self.__get_full_path(tablename=tablename)

        colnew = str(colnew).strip()
        try:
            self.__acquire_lock()
            df = pd.read_csv(tablename, index_col=False)
            if colnew in df.columns:
                self.logger.info('Column already exists "' + str(colnew) + '"')
                return False
            df[colnew] = default_value
            df[colnew] = df[colnew].astype(dtype=data_type)
            df.to_csv(tablename, index=False)
            self.logger.info('Column successfully added "' + str(colnew) + '" as type "' + str(data_type) + '"')
        except Exception as ex:
            self.logger.error('Error occurred: ' + str(ex))
        finally:
            self.__release_lock()

    # When csv writes a numpy array into file, it becomes a string, e.g.
    # '[ 2.32330292e-01 -6.20948195e-01  8.89505982e-01  6.19655311e-01\n -1.97012693e-01... ]'
    def convert_csv_string_array_to_float_array(
            self,
            string_array,
            custom_chars_remove = (),
    ):
        value = string_array.strip()
        # remove front/end brackets "[" and "]"
        value = re.sub(pattern="[\[\]]", repl="", string=value)
        # replace all tabs, newline, etc with space
        value = re.sub(pattern="[ \t\n\r]", repl=" ", string=value)
        # replace multiple spaces with a single space
        value = re.sub(pattern="[ ]+", repl=" ", string=value)
        # replace other custom characters
        for c in custom_chars_remove:
            value = re.sub(pattern="["+c+"]+", repl="", string=value)

        value = value.strip()
        value = value.split(sep=' ')
        float_array = [float(v) for v in value]
        return float_array


if __name__ == '__main__':
    Env.set_env_vars_from_file(env_filepath=EnvRepo().REPO_DIR + os.sep + '.env.fitxf.math.ut')
    DatastoreInterfaceUnitTest(
        ChildClass = Csv,
        logger = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    ).test(
        tablename = 'csvtest',
    )
    exit(0)

    ds = Csv(
        db_params = DbParams.get_db_params_from_envvars(
            identifier = 'csvtest',
            db_create_tbl_sql = '',
            db_table = DATAPATH,
        ),
        logger = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    )
    ds.connect()
    rows = ds.get_all(key='')
    [print(r) for r in rows]

    new_records = [
        {'id': 100, 'text': "give me today shows", 'answer': ' today_showing', 'val': 9.8},
        {'id': 101, 'text': "davai menya kino sigodnya", 'answer': ' today_showing', 'val': 1.2},
        {'id': 102, 'text': "remain text", 'answer': 'random'},
        {'id': 103, 'text': "missing column answer"},
    ]
    ds.add(records=new_records)
    new_records = [
        {'id': 104, 'text': "missing column answer 2", 'no_such_column': 'asdf'},
    ]
    ds.add(records=new_records)

    print('After ADD...')
    [print(r) for r in ds.get_all(key='')]

    for id in [101]:
        ds.delete(match_phrase={'id': id})
        print('After DELETE ' + str(id) + '..')
        rows = ds.get_all(key='')
        [print(r) for r in rows]

    print('Final records')
    [print(r) for r in ds.get_all(key='')]
