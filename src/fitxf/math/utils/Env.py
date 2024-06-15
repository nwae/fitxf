import os
import re
from fitxf.math.utils.File import FileUtils


class Env:

    @staticmethod
    def get_directory_separator():
        return '/' if os.name not in ['nt'] else '\\'

    @staticmethod
    def get_home_dir():
        return os.path.expanduser("~")

    @staticmethod
    def get_home_download_dir():
        dir = str(Env.get_home_dir()) + Env.get_directory_separator() + 'Downloads'
        if not os.path.isdir(dir):
            os.mkdir(dir)
        return dir

    @staticmethod
    def set_env_vars_from_file(env_filepath):
        assert os.path.isfile(env_filepath), 'Not a file "' + str(env_filepath) + '"'
        env_lines = FileUtils(filepath=env_filepath).read_text_file()
        env_lines_cleaned = [line for line in env_lines if line.strip()]
        env_lines_cleaned = [line for line in env_lines_cleaned if not re.match(pattern="^#", string=line)]
        for line in env_lines_cleaned:
            varname, value = line.split(sep="=", maxsplit=1)
            os.environ[varname] = value
            print('Set env var ' + str(varname) + ' = "' + str(value) + '"')


if __name__ == '__main__':
    print(Env.get_home_dir())
    print(Env.get_home_download_dir())
    print(Env.get_directory_separator())
