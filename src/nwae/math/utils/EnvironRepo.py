import logging
import os
import re


class EnvRepo:

    # Relative directory of model download folder
    MODEL_FOLDER_VERSIONS_DIR = '_models/__versions'

    CLOUD_SERVERS = []

    def __init__(
            self,
            # will be given priority over repo_dir if passed in
            user = None,
            repo_dir = None,
            model_version = 'latest',
            logger = None,
    ):
        self.user = user
        self.repo_dir = repo_dir
        self.model_version = model_version
        self.logger = logger if logger is not None else logging.getLogger()

        try:
            self.sysname = os.uname().sysname
        except:
            self.sysname = None

        try:
            self.nodename = os.uname().nodename
        except:
            self.nodename = None

        try:
            import google.colab
            self.in_google_colab = True
            self.logger.info('Detected Colab environment')
        except:
            self.in_google_colab = False

        self.logger.info(
            'Environment sysname "' + str(self.sysname) + '", nodename "' + str(self.nodename)
            + '", in Google Colab "' + str(self.in_google_colab) + '"'
        )

        if self.in_google_colab:
            self.REPO_DIR = '/content/drive/My Drive/colab/poc'
        else:
            self.REPO_DIR = self.repo_dir if self.repo_dir is not None else self.guess_repo_dir()
            self.logger.info('Not in any special environment, using repo dir "' + str(self.REPO_DIR) + '"')

        self.logger.info('Set to different environment, REPO_DIR "' + str(self.REPO_DIR))

        self.MODELS_TRAINING_DIR = self.REPO_DIR + r'/data/models/training'
        self.MODELS_PRETRAINED_DIR = \
            self.REPO_DIR + '/' + self.MODEL_FOLDER_VERSIONS_DIR + '/' + self.model_version

        self.CONFIG_DIR = self.REPO_DIR + r'/config'

        # ----- PII -----
        self.CONFIG_REGEX_DIR = self.CONFIG_DIR + r'/pii/regex'
        self.CONFIG_REGEX_EXCEPTIONS_FILEPATH = self.CONFIG_REGEX_DIR + r'/RGX.EXCEPT'
        self.CONFIG_UNITTEST_REGEX_EXCEPTIONS_FILEPATH = self.CONFIG_REGEX_DIR + r'/RGX.EXCEPT_UT'

        self.CONFIG_PRESIDIO_DIRECTORY = self.CONFIG_DIR + r'/pii/presidio'
        self.CONFIG_PRESIDIO_YAML_FILEPATH = self.CONFIG_PRESIDIO_DIRECTORY + r'/recognizers.yaml'
        self.CONFIG_PRESIDIO_DEF_ENTITIES_JSON = self.CONFIG_PRESIDIO_DIRECTORY + r'/entities_default.json'
        # For unit tests
        self.CONFIG_UNITTEST_PRESIDIO_DIRECTORY = self.CONFIG_DIR + r'/pii/presidio/unittest'
        self.CONFIG_UNITTEST_PRESIDIO_YAML_FILEPATH = self.CONFIG_UNITTEST_PRESIDIO_DIRECTORY + r'/ut.recognizers.yaml'
        self.CONFIG_UNITTEST_PRESIDIO_DEF_ENTITIES_JSON = self.CONFIG_UNITTEST_PRESIDIO_DIRECTORY + r'/ut.ent.json'

        # ----- NLP DATASETS -----
        self.NLP_DATASET_DIR = self.REPO_DIR + r'/data/nlp-datasets'
        self.NLP_DATASET_TXTPAIR_DIR = self.NLP_DATASET_DIR + r'/text-pairs'
        self.NLP_DATASET_PARACRAWL_DIR = self.NLP_DATASET_DIR + r'/text-pairs/paracrawl.eu'
        self.NLP_DATASET_TATOEBA_DIR = self.NLP_DATASET_DIR + r'/text-pairs/tatoeba'
        self.NLP_DATASET_PII_DIR = self.NLP_DATASET_DIR + r'/pii'

    def guess_repo_dir(self):
        try:
            repo_dir = os.environ['REPO_DIR']
        except Exception as ex:
            self.logger.info('Failed to get repo directory from env var "REPO_DIR", got exception ' + str(ex))
            cwd = os.getcwd()
            self.logger.info('Try to guess repo dir from cwd "' + str(cwd) + '"')
            # Look "/src/" in Linux or "\src\" in Windows
            repo_dir = re.sub(pattern="(/src/.*)|([\\\\]src[\\\\].*)", repl="", string=cwd)
            print('Repository directory guessed as "' + str(repo_dir) + '"')
        return repo_dir


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print(EnvRepo().REPO_DIR)
