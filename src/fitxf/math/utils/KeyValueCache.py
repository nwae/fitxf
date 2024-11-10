import logging
import threading
from fitxf.math.utils.Profile import Profiling


#
# A key/value cache (bad naming below makes it "object"/"result")
#
class KeyValueCache:

    KEY_OBJECT = 'object'   # key
    KEY_RESULT = 'result'   # value
    KEY_REPEAT_COUNT = 'repeat_count'

    def __init__(
            self,
            cache_name = None,
            cache_size = 1000,
            # How much to remove from unproven cache when full, default is 50%
            rm_prop_when_full = 0.5,
            enable_text_similarity_lookup = True,
            text_similarity_fixed_len = 100,
            logger = None,
    ):
        self.cache_name = cache_name
        self.cache_size = cache_size
        assert self.cache_size > 0
        self.rm_prop_when_full = rm_prop_when_full
        self.enable_text_similarity_lookup = enable_text_similarity_lookup
        self.text_similarity_fixed_len = text_similarity_fixed_len
        self.logger = logger if logger is not None else logging.getLogger()

        self.cache = {}
        self.cache_stats = {'name': self.cache_name, 'total': 0, 'hit_exact': 0, 'hit_similar': 0, 'hit_rate': 0.0}

        self.textdiff = None
        if self.enable_text_similarity_lookup:
            # We cannot put on top of file, as there will be circular import
            from fitxf.math.lang.measures.TextDiff import TextDiff
            self.textdiff = TextDiff(logger=self.logger)
        self.ref_texts_keys = []
        self.ref_texts_chardiff_model = []

        self.__mutex_cache = threading.Lock()
        self.profiler = Profiling(logger=self.logger)

        self.logger.info(
            'Initialized SimpleCache with cache size ' + str(self.cache_size)
            + ', remove proportion when cache full ' + str(self.rm_prop_when_full)
            + ', text similarity lookup enabled = ' + str(self.enable_text_similarity_lookup)
        )
        return

    def update_cache_stats(
            self,
            hit_exact = 0,
            hit_similar = 0,
    ):
        assert hit_exact + hit_similar <= 1, 'Hit exact ' + str(hit_exact) + ', hit similar ' + str(hit_similar)
        # Total always add 1
        self.cache_stats['total'] += 1
        self.cache_stats['hit_exact'] += hit_exact
        self.cache_stats['hit_similar'] += hit_similar

        if self.cache_stats['total'] > 0:
            self.cache_stats['hit_rate'] = \
                (self.cache_stats['hit_exact'] + self.cache_stats['hit_similar']) / self.cache_stats['total']
        # Log stats every now and then
        if self.cache_stats['total'] % 500 == 0:
            self.logger.info('Key Value Cache "' + str(self.cache_name) + '" stats now: ' + str(self.cache_stats))
        return

    def derive_key(
            self,
            object,
    ):
        return str(object)

    # Look for <object> and return RESULT, something like lookup "key" and return "value", just naming problem
    def get_from_cache_threadsafe(
            self,
            object,
            # by default is exact key search. if this value >0, means will do a text similarity search
            similarity_threshold = 0.0,
    ):
        key = self.derive_key(object=object)
        hit_exact, hit_similar = 0, 0

        try:
            self.__mutex_cache.acquire()

            # First we try to look for exact match in proven cache
            if key in self.cache.keys():
                self.cache[key][self.KEY_REPEAT_COUNT] += 1
                self.logger.info(
                    'Found exact match in cache "' + str(key) + '": ' + str(self.cache[key])
                    + ' Text repeat count now ' + str(self.cache[key][self.KEY_REPEAT_COUNT])
                    + ' "' + str(key) + '"'
                )
                hit_exact = 1
                return self.cache[key][self.KEY_RESULT]
            elif similarity_threshold > 0.0:
                if not self.ref_texts_keys:
                    return None
                top_keys, top_distances = self.textdiff.text_similarity(
                    candidate_text = key,
                    ref_text_list = self.ref_texts_keys,
                    candidate_text_model = None,
                    ref_text_model_list = self.ref_texts_chardiff_model,
                    ref_str_len = self.text_similarity_fixed_len,
                    top_k = 5,
                    model = 'chardiff',
                )
                if top_keys:
                    if top_distances[0] <= similarity_threshold:
                        key_similar = top_keys[0]
                        self.logger.info(
                            'Found via similarity search for "' + str(key) + '", a similar key "' + str(key_similar)
                            + '" distance ' + str(top_distances[0])
                        )
                        self.cache[key_similar][self.KEY_REPEAT_COUNT] += 1
                        hit_similar = 1
                        return self.cache[key_similar][self.KEY_RESULT]
                return None
            else:
                return None
        # except Exception as ex:
        #     self.logger.error('Unexpected error: ' + str(ex))
        #     raise Exception(ex)
        finally:
            self.update_cache_stats(hit_exact=hit_exact, hit_similar=hit_similar)
            self.__mutex_cache.release()

    def add_to_cache_threadsafe(
            self,
            object, # key
            result, # value
    ):
        key = self.derive_key(object=object)

        try:
            self.__mutex_cache.acquire()

            is_cache_updated = False

            if len(self.cache) >= self.cache_size:
                cache_tmp = self.cache
                self.cache = {}
                # remove all with no hits first
                for k, v in cache_tmp.items():
                    if v[self.KEY_REPEAT_COUNT] > 0:
                        self.logger.info('Keep key "' + str(k) + '": ' + str(v))
                        self.cache[k] = v
                    else:
                        self.logger.info('Discard key "' + str(k) + '": ' + str(v))
                self.logger.info(
                    'Successfully cleanup up cache keeping only those with hits from length ' + str(len(cache_tmp))
                    + ', remaining items ' + str(len(self.cache))
                )
                is_cache_updated = True

                # If still not hit the desired target
                count_desired = round(self.cache_size * (1 - self.rm_prop_when_full))
                if len(self.cache) > count_desired:
                    # remove first added ones
                    count_thr = len(self.cache) - count_desired
                    # Keep only latest added ones
                    self.cache = {k:v for i,(k,v) in enumerate(self.cache.items()) if i >= count_thr}
                    self.logger.warning(
                        'Further clear hit cache to new size ' + str(len(self.cache)) + ', count thr ' + str(count_thr)
                        + ', max cache size ' + str(self.cache_size)
                    )

            if is_cache_updated:
                self.__update_text_model()

            if key not in self.cache.keys():
                self.cache[key] = {
                    self.KEY_OBJECT: object,
                    self.KEY_RESULT: result,
                    self.KEY_REPEAT_COUNT: 0,
                }
                self.ref_texts_keys.append(key)
                self.ref_texts_chardiff_model.append(
                    self.textdiff.get_text_model_chardiff(
                        text = key,
                        ref_str_len = self.text_similarity_fixed_len,
                    )
                )
            return key
        except Exception as ex:
            self.logger.error(
                'Error when adding key "' + str(key) + '" to cache: ' + str(ex)
            )
            return None
        finally:
            self.__mutex_cache.release()

    def __update_text_model(
            self,
    ):
        assert self.__mutex_cache.locked()

        self.ref_texts_keys = list(self.cache.keys())
        self.ref_texts_chardiff_model = [
            self.textdiff.get_text_model_chardiff(
                text = key,
                ref_str_len = self.text_similarity_fixed_len,
            )
            for key in self.cache.keys()
        ]
        self.logger.info('Successfully updated chardiff model')
        return

    def search_similar_object(
            self,
            text,
    ):
        try:
            self.__mutex_cache.acquire()
            return self.textdiff.text_similarity(
                candidate_text = text,
                ref_text_list = self.ref_texts_keys,
                candidate_text_model = None,
                ref_text_model_list = self.ref_texts_chardiff_model,
                ref_str_len = self.text_similarity_fixed_len,
                top_k = 5,
                model = 'chardiff',
            )
        finally:
            self.__mutex_cache.release()


class KeyValueCacheUnitTest:
    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        p = Profiling()
        data = [
            ('yo bro',           None,            ['yo bro']),
            ('dobriy dyen 1',    None,            ['yo bro', 'dobriy dyen 1']),
            ('kamusta 1',        None,            ['yo bro', 'dobriy dyen 1', 'kamusta 1']),
            ('dobriy dyen 1',    'dobriy dyen 1', ['yo bro', 'dobriy dyen 1', 'kamusta 1']), # exact hit
            # similarity search will return "dobriy dyen 1" above
            ('dobriy dyen 2',    'dobriy dyen 1', ['yo bro', 'dobriy dyen 1', 'kamusta 1', 'dobriy dyen 2']), # similar hit
            # At this point, we already reached unproven cache max size=4, and we clear those with no hits,
            # thus only ['dobriy dyen 1', 'kamusta 1'] remaining. 'kamusta 2' added after
            ('kamusta 2',        'kamusta 1',     ['dobriy dyen 1', 'kamusta 1', 'kamusta 2']), # similar hit
            ('kamusta 1',        'kamusta 1',     ['dobriy dyen 1', 'kamusta 1', 'kamusta 2']), # exact hit
            ('sami ludi 1',      None,            ['dobriy dyen 1', 'kamusta 1', 'kamusta 2', 'sami ludi 1']),
            # At this point, we already reached unproven cache max size=4, and we clear those with no hits,
            # thus only ['kamusta 1', 'sami ludi 1'] remaining. 'kamusta 2' added after
            ('sami ludi 1',      'sami ludi 1',   ['kamusta 1', 'sami ludi 1']), # exact hit
            ('sami ludi 2',      'sami ludi 1',   ['kamusta 1', 'sami ludi 1', 'sami ludi 2']), # similar hit
            ('sami ludi 3',      'sami ludi 1',   ['kamusta 1', 'sami ludi 1', 'sami ludi 2', 'sami ludi 3']), # similar hit
            # at this point cache will clear again keeping only those hit & latest ones added
            ('sami ludi 1',      'sami ludi 1',   ['kamusta 1', 'sami ludi 1']), # exact hit
        ]
        cache = KeyValueCache(
            cache_size = 4,
            rm_prop_when_full = 0.5,
            logger = self.logger,
        )

        count_hit_exact = 0
        count_hit_similar = 0

        for i, tp in enumerate(data):
            p.start_time_profiling(id='ut')

            print('At line #' + str(i+1) + '. ' + str(data[i]) + ', cache state ' + str(cache.cache))

            txt, expected_res_from_cache, expected_proven_cache_keys = tp
            res = cache.get_from_cache_threadsafe(
                object = txt,
                similarity_threshold = 0.2,
            )
            cache.add_to_cache_threadsafe(
                object = txt,
                result = txt,
            )
            if res is not None:
                if res == txt:
                    print('Exact hit with "' + str(res) + '" == "' + str(txt) + '"')
                    count_hit_exact += 1
                else:
                    print('Similar hit with "' + str(res) + '" != "' + str(txt) + '"')
                    count_hit_similar += 1
            # print('added key:', added_key, ', text', txt, ', result', res, ', expected result', expected_result)
            # print('#' + str(i) + '. Proven cache:\n' + str(cache.cache_proven) + ', unproven cache:\n' + str(cache.cache_unproven))
            assert res == expected_res_from_cache, \
                '#' + str(i+1) + '. Get result "' + str(res) + '" not expected result "' \
                + str(expected_res_from_cache) + '" for text "' + str(txt) + '", test data ' + str(tp)
            assert list(cache.cache.keys()) == expected_proven_cache_keys, \
                '#' + str(i+1) + '. Cache keys ' + str(list(cache.cache.keys())) \
                + '" not expected ' + str(expected_proven_cache_keys) + '" for text "' + str(txt) \
                + '", test data ' + str(tp)

            print('(AFTER ADD) At line #' + str(i+1) + '. cache state ' + str(cache.cache))

            top_keys, top_dists = cache.search_similar_object(text=txt)
            print('Sim search "' + str(txt) + '": ' + str(list(zip(top_keys, top_dists))))

            p.record_time_profiling(id='ut', msg=data[i], logmsg=True)

        print(cache.cache_stats)
        assert cache.cache_stats['total'] == len(data), \
            'Data length ' + str(len(data)) + ' but cache total ' + str(cache.cache_stats['total'])
        assert cache.cache_stats['hit_exact'] == count_hit_exact == 4, \
            'Exact hits ' + str(cache.cache_stats['hit_exact'])
        assert cache.cache_stats['hit_similar'] == count_hit_similar == 4, \
            'Similar hits ' + str(cache.cache_stats['hit_similar'])

        print('ALL TESTS PASSED')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    KeyValueCacheUnitTest().test()

    exit(0)
