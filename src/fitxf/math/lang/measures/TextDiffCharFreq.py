import logging
import collections
import numpy as np
from fitxf.math.lang.measures.TextDiffInterface import TextDiffInterface
from fitxf.utils import Logging


#
# The state-of-the-art way comparing 2 texts would be to calculate the cosine distance between
# the embedding of the texts. But this requires loading a big model, and doing slow inference.
# We want something without external dependencies whatsoever, fast, simple, standalone.
#
# We implement a few low-cost, low-resource, simple measures as follows:
#
# 1. CHARACTER FREQUENCY or "charfreq"
#   Slower than chardiff, non-strict character position independent measure.
#   Cannot quickly calculate between a text and a list of texts.
#   Fast implementation of how different are 2 texts. Min 0 (no diff), max 1.0 (100% different)
#   In a way it is using similar concepts along the lines of "Bog of Words", but is "Bag of Characters"
#   instead. Other differences include we don't maintain a vocabulary of characters and do not use
#   cosine similarity.
#   Instead, we implement a totally standalone, no admin management embedding models, vocab,
#   simple & convenient measure, purely between 2 texts.
#
#   Mathematically between 2 texts, charfreq is given by
#     SUM {abs(count_candidate_text(c) - count_ref_text(c))} / SUM{max(character_count(c, candidate_text, ref_text))}
#   over all characters c in both candidate_text and text2
#
#   Measure Consistency & Must Condition:
#     charfreq = 1.0 when no common character is shared (including space). e.g. 'IloveMcD' and 'ЯдюблюМзкД'
#     charfreq = 0.0 when count of characters are equal regardless of order
#
class TextDiffCharFreq(TextDiffInterface):

    MODEL_PARAMS_DEFAULT = {'n_tuple': 3}

    def __init__(
            self,
            log_time_profilings = False,
            logger = None,
    ):
        super().__init__(
            log_time_profilings = log_time_profilings,
            logger = logger,
        )
        return

    def get_model_params(self, n_tuple: int) -> dict:
        return {'n_tuple': n_tuple}

    # Just a simple dictionary of tuple character frequencies
    # E.g. "privet medved", n tuple 3 becomes
    #   {'pri': 1, 'riv': 1, 'ive': 1, 'vet': 1, 'et ': 1, 't m': 1, ' me': 1, 'med': 1, 'edv': 1, 'dve': 1}
    # For n tuple = 2
    #   {'pr': 1, 'ri': 1, 'iv': 1, 've': 2, 'et': 1, 't ': 1, ' m': 1, 'me': 1, 'ed': 1, 'dv': 1}
    # For n tuple = 1
    #   {'p': 1, 'r': 1, 'i': 1, 'v': 2, 'e': 3, 't': 1, ' ': 1, 'm': 1, 'd': 1}
    def get_text_model(
            self,
            text: str,
            model_params = MODEL_PARAMS_DEFAULT,
    ):
        n_tuple = model_params['n_tuple']
        s_consecutive_chars = [text[i:(i+n_tuple)] for i in range(len(text)) if i<len(text) - n_tuple]
        counter_text_model = dict(collections.Counter(s_consecutive_chars))
        self.logger.debug(
            'charfreq model for text "' + str(text) + '", n tuple ' + str(n_tuple) + ': ' + str(counter_text_model)
        )
        return counter_text_model

    def text_difference(
            self,
            candidate_text,
            ref_text_list,
            # option for user to pre-calculate to whichever text model being used
            candidate_text_model = None,
            ref_text_model_list = None,
            model_params: dict = MODEL_PARAMS_DEFAULT,
            top_k = 3,
    ) -> tuple: # returns tuple of top text list & top scores list
        assert len(ref_text_list) > 0, 'No reference to compare with for text "' + str(candidate_text) + '"'

        if ref_text_model_list is not None:
            assert len(ref_text_list) == len(ref_text_model_list), \
                'Diff length text list ' + str(len(ref_text_list)) + ' model list ' + str(len(ref_text_model_list))

        id_timer = 'text_similarity (' + str(self) + ')'
        self.profiler.start_time_profiling(id=id_timer)

        candidate_counter = self.get_text_model(text=candidate_text, model_params=model_params)
        diff_measures = []
        for i, ref_text in enumerate(ref_text_list):
            diff = self.text_difference_charfreq(
                candidate_text_or_model = candidate_counter if candidate_text_model is None else candidate_text_model,
                ref_text_or_model = ref_text if ref_text_model_list is None else ref_text_model_list[i],
                model_params = model_params,
            )
            diff_measures.append(diff)

        diff_measures = np.array(diff_measures)
        closest = np.argsort(diff_measures)
        top_texts, top_dist = [ref_text_list[i] for i in closest], diff_measures[closest].tolist()
        res = top_texts[0:min(top_k, len(top_texts))], top_dist[0:min(top_k, len(top_dist))]

        self.profiler.record_time_profiling(
            id = id_timer,
            msg = '',
            logmsg = self.log_time_profilings,
        )
        return res

    # Mathematically between 2 texts, charfreq is given by
    #   SUM {abs(count_candidate_text(c) - count_ref_text(c))} / SUM{max(character_count(c, candidate_text, ref_text))}
    # over all characters c in both candidate_text and text2
    def text_difference_charfreq(
            self,
            candidate_text_or_model,
            ref_text_or_model,
            model_params: dict = MODEL_PARAMS_DEFAULT,
    ):
        toc = candidate_text_or_model
        counter_candidate = self.get_text_model(text=toc, model_params=model_params) if type(toc) is str else toc
        toc = ref_text_or_model
        counter_ref = self.get_text_model(text=toc, model_params=model_params) if type(toc) is str else toc

        all_unique_keys = list(set(list(counter_candidate.keys()) + list(counter_ref.keys())))
        all_unique_keys_max = {
            k: max(
                counter_candidate.get(k, 0),
                counter_ref.get(k, 0)
            )
            for k in all_unique_keys
        }
        # sum the total of the max frequencies of all tuples
        total_count = np.array([all_unique_keys_max[k] for k in all_unique_keys])
        # self.logger.debug('Total count ' + str(total_count) + ' from map ' + str(all_unique_keys_max))
        diff = [
            abs(
                counter_candidate.get(k, 0) - counter_ref.get(k, 0)
            )
            for k in all_unique_keys
        ]
        np_diff = np.array(diff)
        # self.logger.debug(
        #     'keys:\n'+ str(all_unique_keys) + '\nnp diff:\n' + str(np_diff) + '\ntotal counts:\n' + str(total_count)
        # )
        # diff_map = [(k, v) for k,v in zip(all_unique_keys, diff)]
        diff_measure = np.sum(np_diff) / np.sum(total_count)
        self.logger.debug(
            'Difference measure = ' + str(diff_measure) # + ', difference map ' + str(diff_map)
            + ', keys max map ' + str(all_unique_keys_max)
            + ' for texts "' + str(candidate_text_or_model) + '" and "' + str(ref_text_or_model) + '"'
        )
        return diff_measure


class TextDiffCharFreqUnitTest:
    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        obj = TextDiffCharFreq(log_time_profilings=False, logger=self.logger)

        s = "privet medved"

        # Check charfreq model is correct
        s_rep = obj.get_text_model(text=s, model_params={'n_tuple': 2})
        assert s_rep == {'pr': 1, 'ri': 1, 'iv': 1, 've': 2, 'et': 1, 't ': 1, ' m': 1, 'me': 1, 'ed': 1, 'dv': 1}, \
            'Wrong charfreq model for "' + str(s) + '": ' + str(s_rep)

        # Check charfreq metric is correct
        for t1, t2, expected_diff in (
                # Exactly similar text
                ('exactly same text must be 0.0 measure', 'exactly same text must be 0.0 measure', 0.0),
                # Completely different text
                ('IloveMcD', '난는맥디를좋아', 1.0),
                # Usual cases
                ('1234567890', '12345abcde', 0.615385),
                ('your facebook OTP is 123456', 'your facebook OTP is 123666', 0.142857),
                # diff measure will not be 100% here because of common space ' ' characters
                ('I love McD', '난는 맥디를 좋아', 0.866667),
        ):
            diff = obj.text_difference_charfreq(
                candidate_text_or_model = t1,
                ref_text_or_model = t2,
                model_params = {'n_tuple': 1},
            )
            self.logger.info(
                'Charfreq between "' + str(t1) + '" & "' + str(t2) + '": ' + str(diff)
            )
            assert round(diff, 6) == expected_diff, \
                'Diff ' + str(diff) + ' not equal to expected diff ' + str(expected_diff)

        # Check both metric is correct by top_k
        test_text_list = ['test test', 'hi how are u', 'how are you', '****asdf 0123456789 0123456789', 'how are']
        close_texts, close_scores = obj.text_difference(
            candidate_text = 'hi how are you',
            ref_text_list = test_text_list,
            top_k = 4,
            model_params = {'n_tuple': 1},
        )
        # exp_scores = [0.14285714285714285, 0.21428571428571427, 0.5, 0.9047619047619048]
        exp_scores = [0.15384615384615385, 0.23076923076923078, 0.5384615384615384, 0.8947368421052632]
        assert close_texts == ['hi how are u', 'how are you', 'how are', 'test test'], \
            'Close texts ' + str(close_texts)
        assert close_scores == exp_scores,  'Close scores ' + str(close_scores) + ' not ' + str(exp_scores)

        self.logger.info('ALL TESTS PASSED')
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    TextDiffCharFreqUnitTest(logger=lgr).test()
    exit(0)
