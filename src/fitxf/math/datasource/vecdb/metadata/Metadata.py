import json
import logging
import os
import threading
import numpy as np
from datetime import datetime
from fitxf import MetadataInterface
from fitxf import DbParams, Datastore, DbLang
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Env import Env


class ModelMetadata(MetadataInterface):

    def __init__(
            self,
            # name to identify which user/table/etc this metadata is referring to
            user_id,
            metadata_tbl_name = 'model_metadata',
            logger = None,
    ):
        super().__init__(
            user_id = user_id,
            metadata_tbl_name = metadata_tbl_name,
            logger = logger,
        )

        self.db_params_metadata = DbParams.get_db_params_from_envvars(
            identifier = str(self.__class__),
            # fill in later below
            db_create_tbl_sql = None,
            db_table = self.metadata_tbl_name,
            verify_certs = os.environ.get("VERIFY_CERTS", "1").lower() in ['1', 'true', 'yes'],
        )

        # For metadata
        # Change create table syntax
        if self.db_params_metadata.db_type in ('mysql',):
            self.db_params_metadata.db_create_table_sql = DbLang.get_db_syntax_create_table_mysql(
                tablename = "<TABLENAME>",
                columns = [
                    "`" + str(self.COL_METADATA_USERID) + "` varchar(255) NOT NULL",
                    "`" + str(self.COL_METADATA_IDENTIFIER) + "` varchar(255) NOT NULL",
                    "`" + str(self.COL_METADATA_TIMESTAMP) + "` double DEFAULT NULL",
                    "`" + str(self.COL_METADATA_VALUE) + "` varchar(5000) DEFAULT NULL",
                ],
            )
        # + ', PRIMARY KEY (' + str(self.COL_METADATA_INDEX) + ', ' + str(self.COL_METADATA_IDENTIFIER) + ')' \
        self.logger.info(
            'Using DB create table sql syntax as "' + str(self.db_params_metadata.db_create_table_sql)
            + '" for DB type "' + str(self.db_params_metadata.db_type) + '"'
        )

        # Model params, last update times, etc.
        self.db_metadata = Datastore(
            db_params = self.db_params_metadata,
            logger = self.logger,
        ).get_data_store()
        self.logger.info('Connected to underlying metadata DB ' + str(self.db_params_metadata.get_db_info()))

        self.__mutex_db = threading.Lock()
        self.last_cleanup_time = datetime(year=2000, month=1, day=1)
        self.min_interval_secs_cleanup = 30
        return

    def __get_specific_metadata(
            self,
            metadata_identifier,
    ):
        try:
            self.__mutex_db.acquire()

            match_phrase = {
                self.COL_METADATA_USERID: self.user_id,
                self.COL_METADATA_IDENTIFIER: metadata_identifier,
            }
            rows = self.db_metadata.get(
                match_phrase = match_phrase,
                tablename = self.db_params_metadata.db_table,
            )
            if len(rows) == 0:
                raise Exception('No metadata returned for ' + str(match_phrase) + '. Returned rows ' + str(rows))
            elif len(rows) == 1:
                return rows[0]
            else:
                last_timestamp = -np.inf
                row_keep = None
                for r in rows:
                    timestamp = r[self.COL_METADATA_TIMESTAMP]
                    if timestamp > last_timestamp:
                        row_keep = r

                self.logger.debug(
                    'Metadata returned > 1 rows from table/index "' + str(self.db_params_metadata.db_table)
                    + '": ' + str(rows) + ', keep: ' + str(row_keep)
                )
                return row_keep
        except Exception as ex:
            self.logger.error(
                'Error getting metadata "' + str(metadata_identifier) + '", returning "' + str(None) + '": ' + str(ex)
            )
        finally:
            self.__mutex_db.release()

    def get_metadata(
            self,
            identifier,
    ):
        try:
            row = self.__get_specific_metadata(
                metadata_identifier = identifier,
            )
            value = row[self.COL_METADATA_VALUE]
            try:
                row[self.COL_METADATA_VALUE] = json.loads(value)
            except:
                pass
            return row
        except Exception as ex:
            self.logger.error('Error get metadata identifier "' + str(identifier) + '": ' + str(ex))
            return None

    # signify that model has been updated
    def update_metadata_identifier_value(
            self,
            identifier: str,
            value: str,
    ):
        if type(value) is dict:
            self.logger.warning(
                'Value for metadata identifier "' + str(identifier)
                + '" is not str but dict. Will try to convert value: ' + str(value)
            )
            try:
                value_str = json.dumps(value)
            except Exception as ex:
                self.logger.error(
                    'Failed to convert value for metadata identifier "' + str(identifier) + '", value ' + str(value)
                    + ', exception: ' + str(ex)
                )
                value_str = str(value)
        else:
            value_str = value
        insert_records = [
            {
                self.COL_METADATA_USERID: self.user_id,
                self.COL_METADATA_IDENTIFIER: identifier,
                self.COL_METADATA_TIMESTAMP: self.get_timestamp(),
                self.COL_METADATA_VALUE: value_str,
            }
        ]
        self.logger.debug('Try to update metadata with records: ' + str(insert_records))
        return self.__update_metadata_to_db(
            tablename = self.db_params_metadata.db_table,
            records = insert_records,
        )

    def __update_metadata_to_db(
            self,
            tablename,
            records,
    ):
        # add timestamp to records
        for r in records:
            r[self.COL_METADATA_TIMESTAMP] = self.get_timestamp()

        try:
            self.__mutex_db.acquire()
            tdif_cleanup = datetime.now() - self.last_cleanup_time
            tdif_cleanup_secs = tdif_cleanup.days * 86400 + tdif_cleanup.seconds + tdif_cleanup.microseconds / 1000000
            if tdif_cleanup_secs > self.min_interval_secs_cleanup:
                for mp in [
                    {
                        self.COL_METADATA_USERID: d[self.COL_METADATA_USERID],
                        self.COL_METADATA_IDENTIFIER: d[self.COL_METADATA_IDENTIFIER]
                    } for d in records
                ]:
                    try:
                        # Table might not exist if first time
                        res = self.db_metadata.delete(
                            tablename = tablename,
                            match_phrase = mp,
                        )
                        self.logger.info(
                            'Deleted from metadata table using match phrase ' + str(mp) + ', result ' + str(res)
                        )
                    except Exception as ex:
                        # For
                        self.logger.error('Error deleting metadata with match phrase "' + str(mp) + '": ' + str(ex))
                    # update last cleanup time
                    self.last_cleanup_time = datetime.now()
            else:
                self.logger.info('Ignore metadata cleanup, last done only ' + str(tdif_cleanup_secs) + ' ago')

            self.db_metadata.add(
                tablename = tablename,
                records = records,
            )
            self.logger.info(
                'Successfully wrote metadata records to "' + str(tablename) + '": '
                + str([{k: str(v)[0:min(300,len(str(v)))] for k, v in r.items()} for r in records])
            )
            return records
        finally:
            self.__mutex_db.release()

    def cleanup(
            self,
    ):
        try:
            self.__mutex_db.acquire()
            res = self.db_metadata.delete(
                match_phrase = {self.COL_METADATA_USERID: self.user_id},
                tablename = self.db_params_metadata.db_table,
            )
            self.logger.info('Successfully deleted metadata for index "' + str(self.user_id) + '": ' + str(res))
            return {'deleted': res['deleted']}
        except Exception as ex:
            self.logger.info('Error delete metadata for index "' + str(self.user_id) + '": ' + str(ex))
        finally:
            self.__mutex_db.release()


if __name__ == '__main__':
    er = Env()
    Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env.fitxf.math.ut.mysql')
    md = ModelMetadata(
        user_id = 'test',
        logger = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    )
    for id, val in [('test', '123'), ('test_dict', {'name': 'jane', 'age': 55})]:
        res = md.update_metadata_identifier_value(
            identifier = id,
            value = val,
        )
        print('Wrote id ' + str(id) + ', val ' + str(val) + ', res: ' + str(res))
    exit(0)
