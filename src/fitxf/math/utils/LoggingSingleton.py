import logging
from fitxf.utils import Logging, Singleton


class LoggingSingleton:

    LOGGER_TYPE_ROTATING_FILE = 'rotating_file'
    LOGGER_TYPE_STD = 'std'

    def __init__(
            self,
            logger_type: str = LOGGER_TYPE_STD,
            logger_name: str = Logging.LOGGER_NAME_DEFAULT,
            filepath: str = None,
            log_level = logging.INFO,
            log_format: str = Logging.LOG_FORMAT_DEFAULT,
            file_max_bytes: int = (1048576 * 100),
            file_backup_count: int = 7,
            # do not propagate to other loggers (prevent duplicate logging) if False
            propagate = False,
            stream = 'stdout',
    ):
        assert logger_type in (self.LOGGER_TYPE_STD, self.LOGGER_TYPE_ROTATING_FILE,)
        if logger_type == self.LOGGER_TYPE_ROTATING_FILE:
            self.logger = Logging.get_logger_rotating_file(
                logger_name = logger_name,
                log_level = log_level,
                filename = filepath,
                log_format = log_format,
                max_bytes = file_max_bytes,
                backup_count = file_backup_count,
                propagate = propagate,
            )
        else:
            self.logger = Logging.get_default_logger(
                log_level = log_level,
                propagate = propagate,
                stream = stream,
            )
        return

    def info(self, msg):
        return self.logger.info(msg)

    def debug(self, msg):
        return self.logger.debug(msg)

    def warning(self, msg):
        return self.logger.warning(msg)

    def warn(self, msg):
        return self.logger.warn(msg)

    def error(self, msg):
        return self.logger.error(msg)

    def critical(self, msg):
        return self.logger.critical(msg)

    @staticmethod
    def get_singleton_logger(
            logger_type: str = LOGGER_TYPE_STD,
            logger_name: str = Logging.LOGGER_NAME_DEFAULT,
            filepath: str = None,
            log_level = logging.INFO,
            log_format: str = Logging.LOG_FORMAT_DEFAULT,
            file_max_bytes: int = (1048576 * 100),
            file_backup_count: int = 7,
            # do not propagate to other loggers (prevent duplicate logging) if False
            propagate = False,
            stream = 'stdout',
    ):
        key_id = 'logger_type=' + str(logger_type) + ';filepath=' + str(filepath) + ';log_level=' + str(log_level) \
                 + ';propagate=' + str(propagate) + ';stream=' + str(stream)
        sgt = Singleton(
            class_type = LoggingSingleton,
        ).get_singleton(
            key_id,
            logger_type,
            logger_name,
            filepath,
            log_level,
            log_format,
            file_max_bytes,
            file_backup_count,
            propagate,
            stream,
        )
        return sgt


if __name__ == '__main__':
    lgr = LoggingSingleton.get_singleton_logger()
    lgr.info('Ok')
    exit(0)
