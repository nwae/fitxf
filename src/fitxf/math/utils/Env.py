import os
import re
import logging
from fitxf.math.utils.File import FileUtils
from inspect import getsourcefile


class Env:

    @staticmethod
    def get_source_code_path():
        path = os.path.abspath(getsourcefile(lambda: 0))
        return path

    @staticmethod
    def get_home_dir():
        return os.path.expanduser("~")

    @staticmethod
    def get_home_download_dir():
        dir = str(Env.get_home_dir()) + os.sep + 'Downloads'
        if not os.path.isdir(dir):
            os.mkdir(dir)
        return dir

    @staticmethod
    def set_env_vars_from_file(env_filepath):
        assert os.path.isfile(env_filepath), 'Not a file "' + str(env_filepath) + '"'
        env_lines = FileUtils(filepath=env_filepath).read_text_file()
        env_lines_cleaned = [line for line in env_lines if line.strip()]
        env_lines_cleaned = [line for line in env_lines_cleaned if not re.match(pattern="^#", string=line)]
        print('Set environment variables from file "' + str(env_filepath) + '"')
        for line in env_lines_cleaned:
            varname, value = line.split(sep="=", maxsplit=1)
            os.environ[varname] = value
            print('Set env var ' + str(varname) + ' = "' + str(value) + '"')
        return

    def __init__(
            self,
            model_version = 'latest',
            logger = None,
    ):
        self.model_version = model_version
        self.logger = logger if logger is not None else logging.getLogger()

        try:
            import google.colab
            self.in_google_colab = True
            self.logger.info('Detected Colab environment')
        except:
            self.in_google_colab = False

        self.logger.info('In Google Colab "' + str(self.in_google_colab) + '"')

        if self.in_google_colab:
            self.REPO_DIR = '/content/drive/My Drive/colab/poc'
        else:
            self.REPO_DIR = self.guess_repo_dir(use_cwd_as_ref=True)
            self.logger.info('Not in any special environment, using repo dir "' + str(self.REPO_DIR) + '"')

        self.logger.info('Set to different environment, REPO_DIR "' + str(self.REPO_DIR))

        self.MODELS_PRETRAINED_DIR = self.REPO_DIR + '/_models/' + self.model_version
        self.CONFIG_DIR = self.REPO_DIR + r'/config'

        # ----- NLP DATASETS -----
        self.NLP_DATASET_DIR = self.REPO_DIR + r'/data/nlp-datasets'

    def guess_repo_dir(
            self,
            # Use current working directory as reference is more accurate since external non-related
            # projects can use this code and guess their directories correctly.
            # Otherwise if use source code, will only be correct for internal projects.
            use_cwd_as_ref = True,
    ):
        try:
            repo_dir = os.environ['REPO_DIR']
        except Exception as ex:
            self.logger.info('Failed to get repo directory from env var "REPO_DIR", got exception ' + str(ex))
            src_code_path = os.getcwd() if use_cwd_as_ref else self.get_source_code_path()
            self.logger.info('Try to guess repo dir from cwd "' + str(src_code_path) + '"')
            # Look "/src/" in Linux or "\src\" in Windows
            repo_dir = re.sub(pattern="(/src/.*)|([\\\\]src[\\\\].*)", repl="", string=src_code_path)
            print('Repository directory guessed as "' + str(repo_dir) + '"')
        return repo_dir


if __name__ == '__main__':
    print(Env.get_source_code_path())
    print(Env.get_home_dir())
    print(Env.get_home_download_dir())
    print(Env().REPO_DIR)
    exit(0)
