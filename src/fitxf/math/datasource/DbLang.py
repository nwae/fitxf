import logging
from fitxf.math.utils.Lock import Logging
from fitxf.math.utils.Env import Env


class DbLang:

    def __init__(
            self,
            logger: Logging = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    @staticmethod
    def get_db_syntax_create_table_mysql(
            tablename: str,
            # ["`name` VARCHAR(255) NOT NULL", ...]
            columns: list,
    ):
        assert len(columns) > 0
        sql = "CREATE TABLE `" + str(tablename) + "` ("
        edge = len(columns) - 1
        for i, col_str in enumerate(columns):
            sql += '\n   ' + str(col_str)
            sql += "," if i != edge else ""
        sql += '\n)'
        return sql

    def get_db_syntax_create_table(
            self,
            db_type: str,
            tablename: str,
            columns: list,
    ):
        if db_type == 'mysql':
            return self.get_db_syntax_create_table_mysql(tablename=tablename, columns=columns)
        else:
            return None


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    er = Env(logger=lgr)
    Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env.fitxf.math.ut.mysql')

    s = DbLang(logger=lgr).get_db_syntax_create_table(
        db_type = 'mysql',
        tablename = "<TABLENAME>",
        columns = [
            "`<CONTENT>` TEXT NOT NULL",
            "`<LABEL_USER>` varchar(255) NOT NULL",
            "`<LABEL_NUMBER>` int NOT NULL",
            "`<ENCODING_B64>` TEXT NOT NULL",
        ],
    )
    print(s)
    exit(0)
