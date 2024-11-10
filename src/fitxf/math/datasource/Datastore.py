import logging
import re
from fitxf.math.datasource.DatastoreInterface import DbParams
from fitxf.math.datasource.Csv import Csv
try:
    from fitxf.math.datasource.MySql import MySql
except Exception as __ex_import:
    print('Error importing library: ' + str(__ex_import))
from fitxf.math.utils.Env import Env


class Datastore:

    def __init__(
            self,
            db_params: DbParams,
            logger = None,
    ):
        self.db_params = db_params
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def get_data_store(
            self,
    ):
        if self.db_params.db_type == 'csv':
            assert not re.match(pattern="/", string=self.db_params.db_table), \
                'Must not contain full path in table name or index "' + str(self.db_params.db_table) + '"'
            DbClass = Csv
        elif self.db_params.db_type == 'mysql':
            DbClass = MySql
        else:
            raise Exception('Not supported data store type "' + str(self.db_params.db_type) + '"')

        self.logger.info(
            'Try to instantiate DB class with db params ' + str(self.db_params.get_db_info()) + ''
        )
        return DbClass(
            db_params = self.db_params,
            logger = self.logger,
        )


if __name__ == '__main__':
    Env.set_env_vars_from_file(env_filepath=Env().REPO_DIR + '/.env.fitxf.math.ut')
    dbp = DbParams.get_db_params_from_envvars(identifier='test', db_create_tbl_sql='', db_table='test_table')
    db = Datastore(db_params=dbp)
    db.get_data_store()
    exit(0)
