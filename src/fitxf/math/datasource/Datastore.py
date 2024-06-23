import logging
import re
from fitxf.math.datasource.DatastoreInterface import DbParams
from fitxf.math.datasource.Csv import Csv
from fitxf.math.utils.Env import Env
from fitxf.math.utils.EnvironRepo import EnvRepo


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
            return Csv(
                db_params = self.db_params,
                logger = self.logger,
            )
        else:
            raise Exception('Not supported data store type "' + str(self.db_params.db_type) + '"')


if __name__ == '__main__':
    Env.set_env_vars_from_file(env_filepath=EnvRepo().REPO_DIR + '/.env.fitxf.math.ut')
    dbp = DbParams.get_db_params_from_envvars(identifier='test', db_create_tbl_sql='', db_table='test_table')
    db = Datastore(db_params=dbp)
    db.get_data_store()
    exit(0)
