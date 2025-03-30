import logging
import numpy as np
import math
from fitxf.math.lang.cache.TextSimilarityCache import TextSimilarityCache
from fitxf.utils import Logging, Profiling


class CacheTest:

    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        self.prf = Profiling(logger=self.logger)
        return

    def test(self):
        ref_texts = [
            'The Chinese ate the dimsum by the street in Beijing',
            'The American gulped down the burger',
            'The French was buying a croissant in the local bakery by the river',
            'The Korean bought a pack of kimchi',
            'The Japanese paid for ramen with a 1000 yen bill to the robot machine',
            'The Pinoys loved their local Inasal chicken with Dinuguang',
            'The Honkies sell very expensive dimsum and the aircons keep dripping water by the walk pavements',
        ] * math.floor(1000 / 7)
        random_numbers = np.round(np.random.rand(len(ref_texts)), decimals=10).tolist()
        ref_texts = [t+' '+str(r) for t, r in zip(ref_texts, random_numbers)]
        test_texts = [
            'Tomorrow is a holiday',
            'The Chinese ate the dimsum by the street',
        ] * 100

        cache = TextSimilarityCache(
            cache_name = 'unit_test',
            cache_size = 1000,
            clear_cache_method = 'old',
            rm_prop_when_full = 0.5,
            text_similarity_fixed_len = 100,
            logger = self.logger,
        )

        for t in ref_texts:
            cache.add_to_cache_threadsafe(object=t, result={'txt': t, 'len': len(t)})
        self.logger.info(
            'Added to cache total objects ' + str(len(ref_texts)) + ', cache size ' + str(cache.get_cache_size())
        )

        start_time = self.prf.start()
        diff_thr = 0.4
        hits = 0
        misses = 0
        for t in test_texts:
            res = cache.get_from_cache_threadsafe(object=t, difference_threshold=diff_thr)
            if res:
                hits += 1
            else:
                misses += 1
            # self.logger.info('For object "' + str(t) + '", dif thr ' + str(diff_thr) + ' got ' + str(res))
        diff_secs = self.prf.get_time_dif_secs(start=start_time, stop=self.prf.stop())
        rps = round(len(ref_texts) / diff_secs, 2)
        self.logger.info(
            'Tested get from cache total ' + str(len(test_texts)) + ', took ' + str(diff_secs)
            + 's, rps ' + str(rps) + ' per sec. Hits ' + str(hits) + ', misses ' + str(misses)
        )
        assert hits == misses == 100
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    CacheTest(lgr).test()
    exit(0)
