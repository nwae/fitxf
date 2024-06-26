import warnings
import uuid
import os
import re
from fitxf.utils import Logging, Profiling, Env
#----------------------------------------------------------------------------------
# IC Section
#----------------------------------------------------------------------------------
# Math
from fitxf.math.fit.transform.FitXformInterface import FitXformInterfaceUnitTest
from fitxf.math.fit.transform.FitXformUnitTest import FitXformUnitTest
from fitxf.math.fit.utils.FitUtilsUnitTest import FitUtilsUt
from fitxf.math.fit.utils.TensorUtils import TensorUtilsUnitTest
from fitxf.math.fit.cluster.ClusterUnitTest import ClusterUnitTest
from fitxf.math.fit.cluster.ClusterCosineUT import ClusterCosineUnitTest
# Utils
from fitxf.math.utils.Lock import LockUnitTest
from fitxf.math.utils.ObjPers import UnitTestObjectPersistence
from fitxf.math.utils.Singleton import SingletonUnitTest
from fitxf.math.utils.StringVar import StringVarUnitTest
# Algo
from fitxf.math.algo.encoding.Base64 import Base64UnitTest


class RepoUnitTest:

    def __init__(
            self,
    ):
        self.env_repo = Env()

        self.keys_dir = self.env_repo.REPO_DIR + '/keys'
        self.lm_cache_folder = self.env_repo.MODELS_PRETRAINED_DIR
        self.document_folder = self.env_repo.REPO_DIR + '/data/sample_docs'
        self.tmp_dir = os.environ["TEMP_DIR"]
        self.logger = Logging.get_logger_from_env_var()

        rand_str = re.sub(pattern=".*[\-]", repl="", string=str(uuid.uuid4()))
        self.db_test_table_or_index = 'nwae-math.repo-unit-test.' + str(rand_str)

        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        return

    def test(self):
        profiler = Profiling(logger=self.logger)
        test_record = {}

        t00 = profiler.start()

        for cls in [
            # Math
            FitXformInterfaceUnitTest, FitXformUnitTest, FitUtilsUt, TensorUtilsUnitTest, # HomomorphismUnitTest,
            ClusterUnitTest, ClusterCosineUnitTest,
            # Utils
            LockUnitTest, SingletonUnitTest, StringVarUnitTest,
            UnitTestObjectPersistence,
            # Datastore
            # This test can only run if you already set up Elasticsearch locally
            # MemoryCacheUnitTest, MySqlUnitTest, VecDbUnitTest, VecDbCcrcyTest, VecDbSingletonUnitTest,
            # Algo
            Base64UnitTest, # SortColumnsUnitTest, SortRangesAndOverlapsUnitTest,
            # Models
            # MultiTreeUnitTest,
            # Language
            # LangCharUnitTest, LangUnitTest,
            # Text
            # TextDiffUnitTest,
            # MaskTextSortedUnitTest, AnonymizerUnitTest, RegexPpUnitTest, TxtPreprocessorUnitTest,
            # Language Models
            # LangModelInterfaceTest, LangModelPtUnitTest,
            # Intent
            # ClassifyWrapperUnitTest,
            # InfoRetrievalUnitTest,
        ]:
            t0 = profiler.start()
            # if cls not in [VecDbCcrcyTest]:
            #     continue

            if cls == FitUtilsUt:
                self.logger.info('BEGIN TESTING ' + str(cls))
                ut = FitUtilsUt(logger=self.logger)
                ut.test_map_to_connbr()
                # TODO Uncomment these when we use them
                # ut.test_nn(epochs=5000, plot_graph=False, tol_dif=0.1)
                # ut.test_dist()
            elif cls == FitXformUnitTest:
                self.logger.info('BEGIN TESTING ' + str(cls))
                ut = FitXformUnitTest(
                    lm_cache_folder = self.lm_cache_folder,
                    logger = self.logger,
                )
                ut.test()
            elif cls == TensorUtilsUnitTest:
                self.logger.info('BEGIN TESTING ' + str(cls))
                ut = TensorUtilsUnitTest()
                ut.test_norm()
                ut.test_similarity_cosine_and_similarity_distance()
            else:
                self.logger.info('BEGIN TESTING ' + str(cls))
                cls(logger=self.logger).test()

            t1 = profiler.stop()
            secs_taken = profiler.get_time_dif_secs(start=t0, stop=t1, decimals=10)
            time_taken = str(round(secs_taken, 2))+'s' if secs_taken > 0.01 else \
                    str(round(secs_taken*1000000, 2))+'μs'
            test_record[cls] = {'secs_taken': time_taken}

        self.logger.info('------------------------------------------------------------------------------------')
        self.logger.info('OK DONE ' + str(len(test_record)) + ' TESTS SUCCESSFULLY')
        [
            self.logger.info(
                '   ' + str(i) + '. ' + str(k) + '\n      --> ' + str(v)
            )
            for i, (k, v) in enumerate(test_record.items())
        ]
        self.logger.info('Total secs taken ' + str(profiler.get_time_dif_secs(start=t00, stop=profiler.stop(), decimals=2)))


if __name__ == '__main__':
    env_repo = Env()
    Env.set_env_vars_from_file(env_filepath=env_repo.REPO_DIR + '/.env.fitxf.math.ut')

    RepoUnitTest().test()
    exit(0)
