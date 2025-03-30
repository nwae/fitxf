import logging
import numpy as np
from fitxf.math.lang.measures.TextDiffInterface import TextDiffInterface
from fitxf.math.lang.measures.TextDiffCharDiff import TextDiffCharDiff
from fitxf.math.lang.measures.TextDiffCharFreq import TextDiffCharFreq
from fitxf.utils import Profiling, Logging


class TextDiffSpeedTest:
    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        td_chardiff = TextDiffCharDiff(log_time_profilings=False, logger=self.logger)
        td_charfreq = TextDiffCharFreq(log_time_profilings=False, logger=self.logger)

        for td, itr, exp_rps, n_ref in [
            # Char freq needs to loop 1-by-1 in n ref sentences, thus the expected RPS is super small
            (td_charfreq, 50, 8.0, 1000),
            # Char diff algo is fast with high expected RPS
            (td_chardiff, 2000, 280.0, 1000),
        ]:
            self.logger.info(
                'Start speed test for ' + str(td) + ', n ref ' + str(n_ref) + ' exp rps ' + str(exp_rps)
                + ', iterations ' + str(itr) + '...'
            )
            self.test_speed(
                td = td,
                itr = itr,
                exp_rps = exp_rps,
                n_ref = n_ref,
            )
        self.logger.info('SPEED TESTS PASSED')
        return

    def test_speed(
            self,
            td: TextDiffInterface,
            itr: int,
            exp_rps: float,
            # how many alphabets in our fake language
            num_lang_chars: int = 88,
            min_sentence_len: int = 50,
            max_sentence_len: int = 150,
            # how many sentences in reference text list
            n_ref: int = 1000,
    ):
        prf = Profiling(logger=self.logger)
        model_prms = {'ref_str_len': max_sentence_len, 'append_ordinal': 0} if type(td) in [TextDiffCharDiff] \
            else {'n_tuple': 3}
        l_avg = int((min_sentence_len + max_sentence_len) / 2)
        alphabets = [' '] \
                    + [chr(x) for x in range(ord('a'), ord('z')+1, 1)]
                    # + [chr(x) for x in range(ord('A'), ord('Z')+1, 1)]
                    # + [chr(x) for x in range(ord('0'), ord('9')+1, 1)] \
                    # + [chr(x) for x in range(ord('а'), ord('я')+1, 1)] \
                    # + [chr(x) for x in range(ord('А'), ord('Я')+1, 1)]
        # self.logger.info('Alphabets: ' + str(alphabets))

        # random text lengths between min_sentence_len to max_sentence_len
        rl = np.round(l_avg + (l_avg * np.random.rand(n_ref) - l_avg/2), decimals=0).astype(np.int32)
        # self.logger.info('Random text lengths ' + str(rl) + ', min ' + str(np.min(rl)) + ', max ' + str(np.max(rl)))

        s_ref = ''.join(np.random.choice(alphabets, l_avg).tolist())
        s_ref_model = td.get_text_model(text=s_ref, model_params=model_prms)
        # self.logger.info('String ref length ' + str(len(s_ref_model)) + ': ' + str(s_ref_model))
        ref_text_list = [
            ''.join(np.random.choice(alphabets, s_len).tolist())
            for s_len in rl
        ]
        ref_text_model_list = [
            td.get_text_model(text=s, model_params=model_prms) for s in ref_text_list
        ]
        # self.logger.info('Ref text list ' + str(ref_text_list))
        # self.logger.info('Ref text list model lengths ' + str([len(v) for v in ref_text_model_list]))

        # Test speed, make sure below some threshold
        self.logger.info('Started speed test for ' + str(td))
        start_time = prf.start()
        for i in range(itr):
            a, b = td.text_similarity(
                candidate_text = s_ref,
                ref_text_list = ref_text_list,
                candidate_text_model = s_ref_model,
                ref_text_model_list = ref_text_model_list,
                top_k = 4,
                model_params = model_prms,
            )
            # self.logger.info(str(a) + ': ' + str(b))
            # raise Exception('asdf')

        diff_secs = prf.get_time_dif_secs(start=start_time, stop=prf.stop(), decimals=8)
        rps = round(itr / diff_secs, 2)
        if rps < exp_rps:
            self.logger.warning(
                'Test class ' + str(type(td)) + ', ' +  str(itr) + ' iterations, FAIL rps = ' + str(rps)
                + ' < ' + str(exp_rps)
            )
        else:
            self.logger.info(
                'Test class ' + str(type(td)) + ', ' +  str(itr) + ' iterations, OK rps = ' + str(rps)
                + ' >= ' + str(exp_rps)
            )
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    TextDiffSpeedTest(logger=lgr).test()
    exit(0)
