import logging
import threading
import os
import pandas as pd
from fitxf.math.lang.measures.TextDiffCharDiff import TextDiffCharDiff
from datetime import datetime, timedelta
from fitxf import DbParams
from fitxf import Datastore as DatastoreMaster
from fitxf.utils import Logging, Profiling, Env


#
# A key/value cache (bad naming below makes it "object"/"result")
#
class TextSimilarityCache:

    KEY_ID = 'key'   # key
    KEY_VALUE = 'value'   # value
    KEY_OBJECT_ORI = 'original_object'
    KEY_REPEAT_COUNT = 'repeat_count'
    KEY_DATETIME = 'datetime'

    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

    MAX_KEY_LEN = 128
    CREATE_TABLE_BY_DB_TYPE = {
        'csv': '',
        'mysql':  """CREATE TABLE `<TABLENAME>` (
               `""" + str(KEY_ID) + """` varchar(""" + str(MAX_KEY_LEN) + """) NOT NULL,
               `""" + str(KEY_VALUE) + """` TEXT NOT NULL,
               `""" + str(KEY_DATETIME) + """` varchar(64) NOT NULL,
               `""" + str(KEY_OBJECT_ORI) + """` TEXT NOT NULL,
               PRIMARY KEY (`""" + str(KEY_ID) +  """`)
            )""",
    }

    def __init__(
            self,
            cache_name: str = None,
            cache_size: int = 1000,
            db_params: DbParams = None,
            # Below variables only for dict type cache
            # allowed values 'old', 'unpopular'
            clear_cache_method: str = 'old',
            # How much to remove from unproven cache when full, default is 50%
            rm_prop_when_full: float = 0.5,
            text_similarity_fixed_len: int = 100,
            logger: Logging = None,
    ):
        self.cache_name = cache_name
        self.cache_size = cache_size
        assert self.cache_size > 0
        self.db_params = db_params
        self.clear_cache_method = clear_cache_method
        assert self.clear_cache_method in ['old', 'unpopular']
        self.rm_prop_when_full = rm_prop_when_full
        self.text_similarity_fixed_len = text_similarity_fixed_len
        self.logger = logger if logger is not None else logging.getLogger()

        self.cache_dict = {}
        if self.db_params is not None:
            self.db_params.db_create_table_sql = self.CREATE_TABLE_BY_DB_TYPE[self.db_params.db_type]
            self.cache_db = DatastoreMaster(db_params=db_params, logger=self.logger).get_data_store()
            self.logger.info('Successfully created cache db from db params: ' + str(self.db_params.get_db_info()))
        self.cache_stats = {'name': self.cache_name, 'total': 0, 'hit_exact': 0, 'hit_similar': 0, 'hit_rate': 0.0}

        self.textdiff = TextDiffCharDiff(
            log_time_profilings = True,
            logger = self.logger,
        )
        self.textdiff_model_prms = self.textdiff.get_model_params(
            ref_str_len = self.text_similarity_fixed_len,
            append_ordinal = 0,
        )
        self.ref_texts_keys = []
        self.ref_texts_chardiff_model = []

        self.__mutex_cache = threading.Lock()
        self.profiler = Profiling(logger=self.logger)

        self.logger.info(
            'Initialized SimpleCache with cache size ' + str(self.cache_size)
            + ', clear cache method "' + str(self.clear_cache_method)
            + '", remove proportion when cache full ' + str(self.rm_prop_when_full)
        )
        return

    def get_cache_size(self):
        try:
            self.__mutex_cache.acquire()
            return len(self.cache_dict)
        finally:
            self.__mutex_cache.release()

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

    def clear_cache(self):
        if self.db_params is not None:
            self.cache_db.delete_all(
                # key = self.KEY_ID,
                tablename = self.db_params.db_table,
            )
            self.logger.info('Cleared cache for db params: ' + str(self.db_params.get_db_info()))
        else:
            self.cache_dict = {}
            self.logger.info('Cleared cache of dict type.')
        return

    def get_cache_keys(self):
        if self.db_params is not None:
            res = self.cache_db.get_all(
                tablename = self.db_params.db_table,
                max_records = 1000000,
            )
            df_tmp = pd.DataFrame.from_records(res)
            df_tmp.sort_values(by=[self.KEY_DATETIME], ascending=True, inplace=True)
            ckeys = [r[self.KEY_ID] for r in df_tmp.to_dict(orient='records')]
            return ckeys
        else:
            return list(self.cache_dict.keys())

    def derive_key(
            self,
            object,
    ):
        obj_key = str(object)
        if len(obj_key) > self.MAX_KEY_LEN:
            # obj_key = hashlib.md5(obj_key.encode(encoding="utf-8")).hexdigest()
            obj_key = obj_key[:self.MAX_KEY_LEN]
        return obj_key

    # Look for <object> and return RESULT, something like lookup "key" and return "value", just naming problem
    def get_from_cache_threadsafe(
            self,
            object,
            # by default is exact key search. if this value >0, means will do a text difference search
            difference_threshold = 0.0,
    ):
        if self.db_params is not None:
            # Clear records older than 1 day
            dt = datetime.now() - timedelta(days=1)
            dt_str = dt.strftime(self.DATETIME_FORMAT)
            del_sql = "DELETE FROM `" + str(self.db_params.db_table) + "` WHERE `" \
                        + str(self.KEY_DATETIME) + "` <= '" + str(dt_str) + "'"
            try:
                res = self.cache_db.delete_by_raw_query(
                    raw_query = del_sql,
                    tablename = self.db_params.db_table,
                )
                self.logger.info('Cleanup cache delete query "' + str(del_sql) + '": ' + str(res))
            except Exception as ex:
                self.logger.error('Failed to delete using sql "' + str(del_sql) + '": ' + str(ex))
                pass

            key = self.derive_key(object=object)
            res = self.cache_db.get(
                match_phrase = {self.KEY_ID: key},
                tablename = self.db_params.db_table,
            )
            self.logger.info(
                'Fetch from cache key "' + str(key) + '": ' + str(res)
            )
            if len(res) == 1:
                return res[0][self.KEY_VALUE]
            else:
                return None
        else:
            return self.get_from_cache_dict_threadsafe(
                object = object,
                difference_threshold = difference_threshold,
            )

    # Look for <object> and return RESULT, something like lookup "key" and return "value", just naming problem
    def get_from_cache_dict_threadsafe(
            self,
            object,
            # by default is exact key search. if this value >0, means will do a text similarity search
            difference_threshold = 0.0,
    ):
        key = self.derive_key(object=object)
        hit_exact, hit_similar = 0, 0

        try:
            self.__mutex_cache.acquire()

            if self.db_params is not None:
                res = self.cache_db.get(
                    match_phrase = {self.KEY_ID: key},
                    tablename = self.db_params.db_table,
                )
                if len(res) > 0:
                    self.logger.info(
                        'Found exact match in cache "' + str(self.db_params.db_type) + '" for key "' + str(key)
                        + '": ' + str(res)
                    )
                    hit_exact = 1
                else:
                    return None
            else:
                # First we try to look for exact match in proven cache
                if key in self.cache_dict.keys():
                    self.cache_dict[key][self.KEY_REPEAT_COUNT] += 1
                    self.logger.info(
                        'Found exact match in cache "' + str(key) + '": ' + str(self.cache_dict[key])
                        + ' Text repeat count now ' + str(self.cache_dict[key][self.KEY_REPEAT_COUNT])
                        + ' "' + str(key) + '"'
                    )
                    hit_exact = 1
                    return self.cache_dict[key][self.KEY_VALUE]
                elif difference_threshold > 0.0:
                    if not self.ref_texts_keys:
                        return None
                    top_keys, top_distances = self.textdiff.text_difference(
                        candidate_text = key,
                        ref_text_list = self.ref_texts_keys,
                        candidate_text_model = None,
                        ref_text_model_list = self.ref_texts_chardiff_model,
                        model_params = self.textdiff_model_prms,
                        top_k = 5,
                    )
                    # self.logger.info('Similarity search for "' + str(key) + '", top distances ' + str(
                    #     top_distances) + ', object ' + str(object))
                    # raise Exception('asdf')
                    if top_keys:
                        if top_distances[0] <= difference_threshold:
                            key_similar = top_keys[0]
                            self.logger.info(
                                'Found via similarity search for "' + str(key) + '", a similar key "' + str(key_similar)
                                + '" distance ' + str(top_distances[0])
                            )
                            self.cache_dict[key_similar][self.KEY_REPEAT_COUNT] += 1
                            hit_similar = 1
                            return self.cache_dict[key_similar][self.KEY_VALUE]
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
        if self.db_params is not None:
            obj_key = self.derive_key(object=object)
            rec = {
                self.KEY_ID: obj_key,
                self.KEY_VALUE: result,
                self.KEY_OBJECT_ORI: str(object),
                self.KEY_DATETIME: datetime.now().strftime(self.DATETIME_FORMAT),
            }
            self.cache_db.delete(
                tablename = self.db_params.db_table,
                match_phrase = {self.KEY_ID: obj_key},
            )
            self.cache_db.add(
                records = [rec],
                tablename = self.db_params.db_table,
            )
        else:
            return self.add_to_cache_dict_threadsafe(
                object = object,
                result = result,
            )

    def add_to_cache_dict_threadsafe(
            self,
            object, # key
            result, # value
    ):
        key = self.derive_key(object=object)

        try:
            self.__mutex_cache.acquire()

            is_cache_updated = False

            if len(self.cache_dict) >= self.cache_size:
                self.logger.info(
                    'Cache full at length ' + str(len(self.cache_dict)) + ' >= ' + str(self.cache_size)
                    + ', clear proportion ' + str(self.rm_prop_when_full) + '.'
                )
                if self.clear_cache_method == 'old':
                    # In this case, just clear oldest, which is FIFO, thus nothing to do, as already done below
                    pass
                else:
                    cache_tmp = self.cache_dict
                    self.cache_dict = {}
                    # remove all with no hits first
                    for k, v in cache_tmp.items():
                        if v[self.KEY_REPEAT_COUNT] > 0:
                            self.logger.info('Keep key "' + str(k) + '": ' + str(v))
                            self.cache_dict[k] = v
                        else:
                            self.logger.info('Discard key "' + str(k) + '": ' + str(v))
                    self.logger.info(
                        'Successfully cleanup up cache keeping only those with hits from length ' + str(len(cache_tmp))
                        + ', remaining items ' + str(len(self.cache_dict))
                    )

                # If still not hit the desired target, clear oldest ones
                count_desired = round(self.cache_size * (1 - self.rm_prop_when_full))
                if len(self.cache_dict) > count_desired:
                    # remove first added ones
                    count_thr = len(self.cache_dict) - count_desired
                    # Keep only latest added ones
                    self.cache_dict = {k:v for i,(k,v) in enumerate(self.cache_dict.items()) if i >= count_thr}
                    self.logger.warning(
                        'Further clear hit cache to new size ' + str(len(self.cache_dict)) + ', count thr ' + str(count_thr)
                        + ', max cache size ' + str(self.cache_size)
                    )
                is_cache_updated = True

            if is_cache_updated:
                self.__update_text_model()

            if key not in self.cache_dict.keys():
                self.cache_dict[key] = {
                    self.KEY_VALUE: result,
                    self.KEY_OBJECT_ORI: object,
                    self.KEY_REPEAT_COUNT: 0,
                    self.KEY_DATETIME: datetime.now()
                }
                self.ref_texts_keys.append(key)
                self.ref_texts_chardiff_model.append(
                    self.textdiff.get_text_model(
                        text = key,
                        model_params = self.textdiff_model_prms,
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

        self.ref_texts_keys = list(self.cache_dict.keys())
        self.ref_texts_chardiff_model = [
            self.textdiff.get_text_model(
                text = key,
                model_params = self.textdiff_model_prms,
            )
            for key in self.cache_dict.keys()
        ]
        self.logger.info('Successfully updated chardiff model')
        return

    def search_similar_object(
            self,
            text,
    ) -> tuple:
        try:
            self.__mutex_cache.acquire()
            return self.textdiff.text_difference(
                candidate_text = text,
                ref_text_list = self.ref_texts_keys,
                candidate_text_model = None,
                ref_text_model_list = self.ref_texts_chardiff_model,
                model_params = self.textdiff_model_prms,
                top_k = 5,
            )
        finally:
            self.__mutex_cache.release()


class TextSimilarityCacheUnitTest:
    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self, additional_tests=()):
        self.test_cache_dict()
        if 'db' in additional_tests:
            self.test_cache_db()
        print('ALL TESTS PASSED')
        return

    def test_cache_db(self):
        data = [
            ('123456789012345678901234567890', '1'),
            ('1234567890123456789012345678901234567890', '2'),
        ]
        db_params = DbParams(
            identifier = 'test',
            db_type = os.environ["DB_TYPE"],
            db_host = os.environ["DB_HOST"],
            db_port = os.environ["DB_PORT"],
            db_username = os.environ["DB_USERNAME"],
            db_password = os.environ["DB_PASSWORD"],
            db_database = os.environ["DB_DATABASE"],
            db_table = "unittest_key_value_cache",
        )
        cache = TextSimilarityCache(
            db_params = db_params,
            cache_size = 4,
            clear_cache_method = 'unpopular',
            rm_prop_when_full = 0.5,
            logger = self.logger,
        )
        cache.clear_cache()
        for i, (obj, val) in enumerate(data):
            cache.add_to_cache_threadsafe(
                object = obj,
                result = val,
            )
            res = cache.get_from_cache_threadsafe(
                object = obj,
            )
            assert res == val, \
                'Test DB cache #' + str(i) + ' for object "' + str(obj) + '", got ' + str(res) + ' not ' + str(val)
        print('ALL CACHE DB TESTS PASSED')
        return

    def test_cache_dict(self):
        p = Profiling(logger=self.logger)
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
        cache = TextSimilarityCache(
            db_params = None,
            cache_size = 4,
            clear_cache_method = 'unpopular',
            rm_prop_when_full = 0.5,
            logger = self.logger,
        )
        cache.clear_cache()

        count_hit_exact = 0
        count_hit_similar = 0

        for i, tp in enumerate(data):
            p.start_time_profiling(id='ut')
            # print('At line #' + str(i+1) + '. ' + str(data[i]) + ', cache state ' + str(cache.cache_db))

            txt, expected_res_from_cache, expected_proven_cache_keys = tp
            res = cache.get_from_cache_threadsafe(
                object = txt,
                difference_threshold = 0.2,
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
            cur_keys = cache.get_cache_keys()
            diff_set = list(set(cur_keys).difference(set(expected_proven_cache_keys)))
            assert diff_set == [], \
                '#' + str(i+1) + '. Cache keys ' + str(cur_keys) \
                + '" not expected ' + str(expected_proven_cache_keys) + '" for text "' + str(txt) \
                + '", test data ' + str(tp)

            # print('(AFTER ADD) At line #' + str(i+1) + '. cache state ' + str(cache.cache_db))
            top_keys, top_dists = cache.search_similar_object(text=txt)
            self.logger.info('Sim search "' + str(txt) + '": ' + str(list(zip(top_keys, top_dists))))

            p.record_time_profiling(id='ut', msg=data[i], logmsg=True)

        self.logger.info(cache.cache_stats)
        assert cache.cache_stats['total'] == len(data), \
            'Data length ' + str(len(data)) + ' but cache total ' + str(cache.cache_stats['total'])
        assert cache.cache_stats['hit_exact'] == count_hit_exact == 4, \
            'Exact hits ' + str(cache.cache_stats['hit_exact'])
        assert cache.cache_stats['hit_similar'] == count_hit_similar == 4, \
            'Similar hits ' + str(cache.cache_stats['hit_similar'])

        # Old test
        # Cache max 4, and remove 1 when full everytime
        cache = TextSimilarityCache(
            cache_size = 4,
            clear_cache_method = 'old',
            rm_prop_when_full = 0.3,
            logger = self.logger,
        )
        for i in range(10):
            cache.add_to_cache_threadsafe(object=i, result=10 * i)
            keys = cache.get_cache_keys()
            self.logger.info('***** ' + str(i) + str(cache.cache_dict))
            keys_expected = [str(v) for v in list(range(i + 1)) if i - v < 4]
            assert keys == keys_expected, 'Keys ' + str(keys) + ' not expected ' + str(keys_expected)

        self.logger.info('ALL CACHE DICT TESTS PASSED')
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    er = Env(logger=lgr)
    Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env.fitxf.math.ut')

    TextSimilarityCacheUnitTest(logger=lgr).test(additional_tests=[])

    exit(0)
