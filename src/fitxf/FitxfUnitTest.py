import warnings
import uuid
import os
import re
from venv import logger

from fitxf.utils import Logging, Profiling, Env
#----------------------------------------------------------------------------------
# IC Section
#----------------------------------------------------------------------------------
# Math
from fitxf.math.fit.transform.FitXformInterface import FitXformInterfaceUnitTest
from fitxf.math.fit.transform.FitXformUnitTest import FitXformUnitTest
from fitxf.math.fit.utils.FitUtilsUnitTest import FitUtilsUt
from fitxf.math.fit.arc.ClassifierArcUnitTest import ClassifierArcUnitTest
from fitxf.math.fit.arc.ClassifierArcRf import ClassifierArcRf
from fitxf.math.fit.utils.TensorUtils import TensorUtilsUnitTest
from fitxf.math.fit.utils.MathUtils import MathUtilsUnitTest
from fitxf.math.fit.utils.PatternSearchUnitTest import PatternSearchUnitTest
from fitxf.math.fit.cluster.ClusterUnitTest import ClusterUnitTest
from fitxf.math.fit.cluster.ClusterCosineUT import ClusterCosineUnitTest
from fitxf.math.graph.GraphUtilsUnitTest import GraphUtilsUnitTest
from fitxf.math.utils.PkgVersion import PkgVersion
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

        pkg_utils = PkgVersion(logger=self.logger)
        verdict, (self.py_maj_ver, self.py_min_ver)= pkg_utils.check_python(version="3.12", return_version=True)

        rand_str_uuid = str(uuid.uuid4())
        if (self.py_maj_ver == 3) and (self.py_min_ver <= 11):
            rand_str = re.sub(pattern=".*[\-]", repl="", string=rand_str_uuid)
        else:
            rand_str = re.sub(pattern=".*[-]", repl="", string=rand_str_uuid)
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
            MathUtilsUnitTest, PatternSearchUnitTest, ClusterUnitTest, ClusterCosineUnitTest,
            GraphUtilsUnitTest,
            ClassifierArcUnitTest,
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

            self.logger.info('BEGIN TESTING ' + str(cls))
            if cls == FitUtilsUt:
                ut = FitUtilsUt(logger=self.logger)
                ut.test_map_to_connbr()
                # TODO Uncomment these when we use them
                # ut.test_nn(epochs=5000, plot_graph=False, tol_dif=0.1)
                # ut.test_dist()
            elif cls == FitXformUnitTest:
                ut = FitXformUnitTest(
                    lm_cache_folder = self.lm_cache_folder,
                    logger = self.logger,
                )
                ut.test()
            elif cls == TensorUtilsUnitTest:
                ut = TensorUtilsUnitTest()
                ut.test_norm()
                ut.test_similarity_cosine_and_similarity_distance()
            elif cls == ClassifierArcUnitTest:
                ut = ClassifierArcUnitTest(child_class=ClassifierArcRf, logger=self.logger)
                ut.test()
            else:
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
