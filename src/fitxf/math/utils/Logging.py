import logging
import logging.handlers
import os
import sys
from datetime import datetime


class Logging:

    LOGGER_NAME_DEFAULT = 'fitxf'
    LOGGER_FILENAME_DEFAULT = 'fitxf'

    LOG_FORMAT_DEFAULT = '%(asctime)s: %(name)s: %(levelname)s: <%(filename)s> "%(funcName)s" line #%(lineno)d:\t%(message)s'

    @staticmethod
    def set_debug_config():
        logging.basicConfig(
            level = logging.DEBUG,
            format = Logging.LOG_FORMAT_DEFAULT,
        )

    @staticmethod
    def __set_log_level(
            logger,
            # can be string or number
            log_level,
    ):
        print('Setting log level to "' + str(log_level) + '"')
        if log_level == 'DEBUG':
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(level=logging.DEBUG)
        elif log_level == 'INFO':
            logging.basicConfig(level=logging.INFO)
            logger.setLevel(level=logging.INFO)
        elif log_level == 'WARN':
            logging.basicConfig(level=logging.WARN)
            logger.setLevel(level=logging.WARN)
        elif log_level == 'ERROR':
            logging.basicConfig(level=logging.ERROR)
            logger.setLevel(level=logging.ERROR)
        else:
            logging.basicConfig(level=log_level)
            logger.setLevel(level=log_level)
        print('Log level to "' + str(logger.getEffectiveLevel()) + '"')
        return

    @staticmethod
    def get_logger_from_env_var():
        log_level = logging.DEBUG if os.environ["LOGGER_LEVEL"].lower() == "debug" else None
        log_level = logging.WARN if os.environ["LOGGER_LEVEL"].lower() == "warn" else log_level
        log_level = logging.ERROR if os.environ["LOGGER_LEVEL"].lower() == "error" else log_level
        log_level = logging.INFO if log_level is None else log_level

        if os.environ["LOGGER_TYPE"] == 'rotatingfile':
            return Logging.get_logger_rotating_file(
                logger_name = os.environ["LOGGER_NAME"],
                log_level = log_level,
                filename = os.environ["LOGGER_FILE_PATH"],
                max_bytes = int(os.environ["LOGGER_MAX_MB"]) * 1048576,
                backup_count = int(os.environ["LOGGER_BACKUP_COUNT"]),
                propagate = False,
            )
        else:
            return Logging.get_default_logger(
                log_level = log_level,
                propagate = False,
            )

    """
    Rotating file logger, so we don't need to manage deletion of old log files.
    It will rotate between the fixed number of files, and overwrite the old data.
    """
    @staticmethod
    def get_logger_rotating_file(
            logger_name,
            log_level,
            filename,
            log_format = LOG_FORMAT_DEFAULT,
            # default 100MB
            max_bytes = (1048576 * 100),
            # means max 7 files to rotate each of size <maxBytes>
            backup_count = 7,
            # do not propagate to other loggers (prevent duplicate logging) if False
            propagate = False,
    ):
        # assert os.path.exists(filename), 'File path "' + str(filename) + '" must exist'
        logger = logging.getLogger(logger_name)
        for h in logger.handlers:
            print('Remove handler "' + str(h.name) + '", type "' + str(type(h)))
            logger.removeHandler(hdlr=h)

        Logging.__set_log_level(
            logger = logger,
            log_level = log_level,
        )
        # why logging library hasHandlers()==True, but inside handlers have nothing?
        print(
            'Handlers: ' + str(logger.handlers) + ' have handlers = ' + str(logger.hasHandlers())
            + ', log level "' + str(logger.getEffectiveLevel()) + '"'
        )

        handler = logging.handlers.RotatingFileHandler(
            filename = filename,
            maxBytes = max_bytes,
            backupCount = backup_count,
        )
        print(
            'Logger "' + str(logger_name) + ' added new RotatingFileHandler with filename "' + str(filename)
            + '", max bytes ' + str(max_bytes) + ', backup ' + str(backup_count)
        )
        formatter = logging.Formatter(log_format)
        handler.setFormatter(fmt=formatter)
        logger.addHandler(hdlr=handler)
        logger.propagate = propagate
        return logger

    @staticmethod
    def get_default_logger(
            log_level = logging.INFO,
            # do not propagate to other loggers (prevent duplicate logging) if False
            propagate = False,
            stream = 'stdout',
    ):
        # assert os.path.exists(filename), 'File path "' + str(filename) + '" must exist'
        logger = logging.getLogger(Logging.LOGGER_NAME_DEFAULT)
        for h in logger.handlers:
            print('Remove handler "' + str(h.name) + '", type "' + str(type(h)))
            logger.removeHandler(hdlr=h)

        Logging.__set_log_level(
            logger = logger,
            log_level = log_level,
        )

        formatter = logging.Formatter(Logging.LOG_FORMAT_DEFAULT)
        consoleHandler = logging.StreamHandler(
            stream = sys.stdout if stream in ('stdout',) else sys.stderr,
        )
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
        logger.propagate = propagate
        return logger


if __name__ == '__main__':
    def make_noise(lg, msg):
        lg.debug(msg)
        lg.info(msg)
        lg.warning(msg)
        lg.error(msg)
        lg.critical(msg)
        lg.fatal(msg)

    name = 'test_logger'
    lgr = Logging.get_logger_rotating_file(
        logger_name = name,
        log_level  = logging.INFO,
        log_format = Logging.LOG_FORMAT_DEFAULT,
        filename = 'test.log.' + str(datetime.now().strftime('%Y-%m-%d')) + '',
        max_bytes = (400),
        backup_count = 2,
    )
    for msg in ['moscow', 'astana']:
        make_noise(lg=lgr, msg=msg)

    lgr = Logging.get_logger_rotating_file(
        logger_name = name,
        log_level  = logging.INFO,
        log_format = Logging.LOG_FORMAT_DEFAULT,
        filename = 'test.log.' + str(datetime.now().strftime('%Y-%m-%d')) + '',
        max_bytes = (1000),
        backup_count = 2,
    )
    make_noise(lg=lgr, msg='melbourne')

    logging.critical('test logging without using logger')

    exit(0)
