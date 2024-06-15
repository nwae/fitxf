import logging
import os
import time
from datetime import datetime


class FileUtils:

    def __init__(
            self,
            filepath,
            logger = None,
    ):
        self.filepath = filepath
        self.logger = logger if logger is not None else logging.getLogger()

        self.last_modified_time = self.get_last_modified_time()
        self.logger.info(
            'File "' + str(self.filepath) + '", last modified time "' + str(self.last_modified_time)
        )
        return

    def update_last_modified_time(self):
        self.last_modified_time = self.get_last_modified_time()

    def get_last_modified_time(self):
        t = os.path.getmtime(self.filepath)
        last_modified_time = datetime.fromtimestamp(t)
        self.logger.debug(str(self.filepath) + ': file last modified time "' + str(last_modified_time))
        return last_modified_time

    def is_file_modified(self):
        t_delta = self.get_last_modified_time() - self.last_modified_time
        diff_secs = t_delta.days*86400 + t_delta.seconds + t_delta.microseconds/1000000
        self.logger.debug(str(self.filepath) + ': difference between last modified times = ' + str(diff_secs) + 's')
        return diff_secs > 0

    def read_text_file(
            self,
            encoding = 'utf-8',
            throw_exception = True,
    ):
        try:
            fh = open(self.filepath, 'r', encoding=encoding)
        except IOError as e:
            errmsg = 'Cannot open file [' + str(self.filepath) + ']. ' + str(e)
            self.logger.error(errmsg)
            if throw_exception:
                raise Exception(errmsg)
            else:
                return []

        lines = [row.rstrip() for row in fh]

        fh.close()
        return lines


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    filepath = 'filetest.txt'
    fu = FileUtils(filepath=filepath)

    while True:
        lns = fu.read_text_file()
        # [print(ln) for ln in lns]
        print(fu.get_last_modified_time())
        modified = fu.is_file_modified()
        if modified:
            break
        time.sleep(5)
