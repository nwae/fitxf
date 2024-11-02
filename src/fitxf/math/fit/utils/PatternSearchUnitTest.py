import logging
import os
import re
import platform
import numpy as np
from fitxf.math.fit.utils.PatternSearch import PatternSearch
from fitxf.math.utils.Pandas import Pandas
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Profile import Profiling


class PatternSearchUnitTest:

    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        patsearch = PatternSearch(logger=self.logger)

        x = np.arange(20) % 10
        for seq, exp_matches in [
            (np.array([1, 2, 3, 4]), np.array([1, 11])),
            (np.array([9, 10, 11]), np.array([9])),
            # (np.array([1, 3, 5]), []),
        ]:
            match_idxs = patsearch.math_utils.match_template(x=x, seq=seq)
            assert np.sum((np.array(match_idxs) - exp_matches)**2) < 0.0000000001, \
                'Match indexes ' + str(match_idxs) + ' not ' + str(exp_matches)

        dth, cth = 0.9, 0.9
        test_data = [
            # No repeats
            (dth, cth, "don't destroy this genuine sentence", ""),
            (dth, cth, "another valid sentence with no LLM hysterics", ""),
            # Have repeats
            (dth, 0.6, "zz zz zz abedeee abedeee abedeeee", "abedeee abedeee abedeee"),
            (1.0, cth, "Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1",
             "Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 "),
            (1.0, cth, "Test 222222 Test 222222 Test 222222 Test 222222 Test 222222 Test 222222 Tes",
             "Test 222222 Test 222222 Test 222222 Test 222222 Test 222222 Test 222222 "),
            (dth, 0.6, "bread rice wine beer wine beer", "wine beer wine beer"),
            (dth, 0.6, "LLM hysterics ||| ||| ||| ||| ||| ||| ||", "||| ||| ||| ||| ||| ||| "),
        ]
        for den_thr, cov_thr, s, exp_repeats in test_data:
            res = patsearch.find_repeat_sequences(
                x = np.array([ord(c) for c in s]),
                min_seq_len = 3,
                density_repeat_thr = den_thr,
                coverage_thr = cov_thr,
                hint_separators = np.array([ord(v) for v in [" "]]),
                stagnant_at_optimal_count_thr = 10,
                string_convert_result = True,
            )
            if res is not None:
                s_front, s_repeats, s_back = res[0]['s_parts']
            else:
                s_repeats = ''
            assert s_repeats == exp_repeats, 'Expected "' + str(exp_repeats) + '", got "' + str(s_repeats) + '"'

        for i, (min_c_len, hint_sep, den_thr, cov_thr, s, exp_res) in enumerate([
            #
            # No repeats
            #
            (8, [".", ":", ";"], 0.5, 0.5, "LLMs go hysterical always", None),
            (8, [".", ":", ";"], 0.5, 0.5, "LLMs go hysterical every time, what nonsense from AI", None),
            #
            # Optimal repeats
            #
            (2, [" "], 0.6, 0.5, "AI LLMs 7778888 LLMs 777---- LLMs 777++++ LLMs 7778888 xx LLMs 777**** LLMs 777^^^^ ...",
             {
                 'seq_len': 8, 'prefix': "LLMs 777", 'indexes': [3, 16, 29, 42, 58, 71],
                 'seq_list': [2, 13, 12, 11, 10, 9, 8],
                 'break_reason': 'density/coverage condition met',
             }),
            #
            # Have repeats
            #
            (8, [".", ":", ";"], 0.6, 0.6, "LLMs go hysterical: LLMs go hysterical: LLMs go hysterical: LLMs g",
             {
                 'seq_len': 19, 'prefix': " LLMs go hysterical", 'indexes': [19, 39], 'seq_list': [8, 20],
                 'break_reason': 'density/coverage condition met',
             }),
            (16, [".", ":", ";"], 0.5, 0.5, "LLMs go hysterical: LLMs go hysterical: LLMs go hysterical: LLMs g",
             {
                 'seq_len': 16, 'prefix': " LLMs go hysteri", 'indexes': [19, 39], 'seq_list': [16],
                 'break_reason': 'density/coverage condition met',
             }),
            (8, [".", ":", ";"], 0.3, 0.5, "LLMs go hy111111: LLMs go hy222222222222: LLMs go hy33333333: LLMs g",
             {
                 'seq_len': 10, 'prefix': " LLMs go h", 'indexes': [17, 41], 'seq_list': [8, 24, 16, 12, 10, 11],
                 'break_reason': 'density/coverage condition met',
             }),
            # With space as hint separator
            (8, [".", ":", ";", " "], 0.9, 0.8, "LLMs go hysterical: LLMs go hysterical: LLMs go hysterical: L",
             {
                 'seq_len': 18, 'prefix': "LLMs go hysterical", 'indexes': [0, 20, 40], 'seq_list': [8, 20],
                 'break_reason': 'density/coverage condition met',
             }),
        ]):
            res = patsearch.find_repeat_sequences(
                x = np.array([ord(c) for c in s]),
                min_seq_len = min_c_len,
                hint_separators = np.array([ord(c) for c in hint_sep]),
                density_repeat_thr = den_thr,
                coverage_thr = cov_thr,
                stagnant_at_optimal_count_thr = 100,
                string_convert_result = True,
            )
            if exp_res is None:
                assert res is None
            else:
                assert res is not None, '#' + str(i) + ' result None for "' + str(s) + '"'
                pfx, sq_len, sq_lst, idxs, br = \
                    res[0]['prefix'], res[0]['seq_len'], res[0]['seq_list'], res[0]['indexes'], res[0]['break_reason']
                exp_pfx, exp_sq_len, exp_sq_lst, exp_idxs, exp_br = \
                    exp_res['prefix'], exp_res['seq_len'], exp_res['seq_list'], exp_res['indexes'], exp_res['break_reason']
                assert re.match(pattern=exp_pfx, string=pfx), \
                    '#' + str(i) + ' Expected prefix "' + str(exp_pfx) + '*" but got "' + str(pfx) + '"'
                assert sq_len >= exp_sq_len, \
                    '#' + str(i) + ' Expected >= seq len ' + str(exp_sq_len) + ' but got ' + str(sq_len)
                # assert sq_lst == exp_sq_lst, \
                #     '#' + str(i) + ' Expected seq list ' + str(exp_sq_lst) + ' but got ' + str(sq_lst)
                assert idxs == exp_idxs, \
                    '#' + str(i) + ' Expected indexes ' + str(exp_idxs) + ' but got ' + str(idxs)
                assert br == exp_br, \
                    '#' + str(i) + ' Expected break reason "' + str(exp_br) + '" but got "' + str(br) + '"'

        long_string = ""
        for i in range(10):
            long_string += " ".join([str(v) for v in np.arange(20).tolist()]) + " <RANDOM>" \
                           + str(np.random.random(5)) + '</RANDOM>.\n'
        long_string = long_string + " ".join([str(v) for v in np.arange(10).tolist()])
        res = patsearch.find_repeat_sequences(
            x = np.array([ord(c) for c in long_string]),
            min_seq_len = 11,
            hint_separators = np.array([ord(c) for c in ["\n"]]),
            stagnant_at_optimal_count_thr = 100,
            string_convert_result = True,
            density_repeat_thr = 0.5,
            coverage_thr = 0.5,
        )
        pfx, idxs, sqlst = res[0]["prefix"], res[0]["indexes"], res[0]["seq_list"]
        assert res[0]["prefix"] in [
            "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 <RANDOM>[0.",
            "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 <RANDOM>[0",
        ], 'Got ' + str(pfx)
        assert res[0]["indexes"] == [0, 125, 250, 375, 500, 625, 750, 875, 1000, 1125], 'Got instead ' + str(idxs)
        # assert res[0]["seq_list"] == [11, 125, 68, 40, 67, 50, 66, 57, 65, 60, 64, 62, 63, 61], 'Got ' + str(sqlst)

        # Quiet during speed test
        self.logger.setLevel(level=logging.ERROR)
        os.environ["ENABLE_SLOW_LOGGING"] = "false"
        obj = PatternSearch(
            logger = self.logger,
        )
        time_records = []
        prf = Profiling(logger=self.logger)
        for i in range(200):
            start = prf.start()
            res = obj.find_repeat_sequences(
                x = np.array([
                    ord(c) for c in "Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1 Test 1"]
                ),
                min_seq_len = 4,
                hint_separators = np.array([ord(c) for c in ["1", " "]]),
                string_convert_result = True,
            )
            ms = round(1000 * prf.get_time_dif_secs(start=start, stop=prf.stop()), 3)
            time_records.append(ms)
            print('#' + str(i) + ': ' + str(res)) if i%100==0 else 1

        # Noisy back
        self.logger.setLevel(level=logging.INFO)
        pf = platform.platform()
        pf_info = pf.split("-")
        processor = pf_info[2]
        if processor in ['x86_64']:
            avg_ms_thr = 32
        else:
            avg_ms_thr = 6
        self.logger.info(
            'Platform processor "' + str(processor) + '", "' + str(pf) + '": ' + str(pf_info)
            + ', avg ms thr ' + str(avg_ms_thr)
        )

        avg_ms = np.mean(np.array(time_records))
        self.logger.info('Average time ' + str(avg_ms) + ', time records: ' + str(time_records))
        assert avg_ms < avg_ms_thr, 'FAIL Average ms ' + str(avg_ms) + ' < ' + str(avg_ms_thr) + 'ms'

        self.logger.info('ALL TESTS PASSED')
        return


if __name__ == '__main__':
    Pandas.increase_display()
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    PatternSearchUnitTest(logger=lgr).test()
    exit(0)
