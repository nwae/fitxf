import logging
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
# CHARACTER DIFFERENCE or chardiff
#    Fastest, strict character dependent measure.
#    Can quickly calculate text against a list of other texts.
#    Sentence represented as a vector of its character unicode values, then appended with 0's to achieve
#    fixed length. Thus fast comparison (single matrix multiplication) can be made to a set of many reference
#    texts.
#    See function text_similarity_chardiff() below for more details.
#
#   Measure Consistency & Must Condition:
#     charfreq = 1.0 when no common character is shared (including space). e.g. 'IloveMcD' and 'ЯдюблюМзкД'
#     charfreq = 0.0 when count of characters are equal regardless of order
#
class TextDiffCharDiff(TextDiffInterface):

    APPEND_ORDINAL_DEFAULT = 0
    MODEL_PARAMS_DEFAULT = {'ref_str_len': 100, 'append_ordinal': APPEND_ORDINAL_DEFAULT}

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

    def get_model_params(self, ref_str_len: int, append_ordinal: int) -> dict:
        return {'ref_str_len': ref_str_len, 'append_ordinal': append_ordinal}

    # Represent each character by its Unicode integer, then append 0's (by default) to a fixed length
    def get_text_model(
            self,
            text,
            model_params = MODEL_PARAMS_DEFAULT,
    ):
        ref_str_len = model_params['ref_str_len']
        append_ordinal = model_params['append_ordinal']
        ordinals_list = [ord(c) for c in text[0:min(ref_str_len, len(text))]] \
                        + [append_ordinal] * max(0, (ref_str_len - len(text)))
        self.logger.debug('chardiff model for text "' + str(text) + '": ' + str(ordinals_list))
        return ordinals_list

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

        res = self.text_difference_chardiff(
            candidate_text = candidate_text,
            ref_text_list = ref_text_list,
            candidate_text_model = candidate_text_model,
            ref_text_model_list = ref_text_model_list,
            model_params = model_params,
            top_k = top_k
        )

        self.profiler.record_time_profiling(
            id = id_timer,
            msg = '',
            logmsg = self.log_time_profilings,
        )
        return res

    # Measures total number of different characters with a list of reference texts, thus faster than above
    # Convert all character to unicode number, take difference, convert to 0 and 1 (non-zero), then sum each
    # row (reference text) divided by fixed string length.
    # Take the lowest value as most similar text.
    def text_difference_chardiff(
            self,
            candidate_text,
            ref_text_list,
            # Option to pass in pre-calculated text model
            candidate_text_model = None,
            ref_text_model_list = None,
            model_params: dict = MODEL_PARAMS_DEFAULT,
            top_k = 3,
    ) -> tuple: # returns tuple of top text list & top scores list
        assert len(ref_text_list) > 0, 'No reference to compare with for text "' + str(candidate_text) + '"'

        if ref_text_model_list is None:
            ref_text_model_list = [
                self.get_text_model(text=s, model_params=model_params) for s in ref_text_list
            ]
        if candidate_text_model is None:
            candidate_text_model = self.get_text_model(text=candidate_text, model_params=model_params)

        candidate_vec = np.array(candidate_text_model, dtype=np.int32)
        candidate_vec_l = np.sum(1 * (candidate_vec > 0))
        ref_matrix = np.array(ref_text_model_list, dtype=np.int32)
        # self.logger.debug(
        #     'Candidate vec:\n' + str(candidate_vec.tolist()) + '\n, ref matrix:\n'
        #     + str([v.tolist() for v in ref_matrix])
        # )
        ref_matrix_l = np.sum(1 * (ref_matrix > 0), axis=-1)
        # Each reference vector will have their new lengths, max of (candidate text, ref text)
        ref_matrix_candidate_l = np.maximum(ref_matrix_l, candidate_vec_l)
        # self.logger.debug(
        #     'Candidate vec length ' + str(candidate_vec_l) + ', ref matrix lengths ' + str(ref_matrix_l)
        #     + ', after maximum of both ' + str(ref_matrix_candidate_l)
        # )

        # Make negative differences positive, we just need absolute value
        diff = np.abs(ref_matrix - candidate_vec)
        # self.logger.debug('Diff ' + str(diff))
        # binary difference only, 0 remains 0, positive values become 1
        diff_binary = 1 * (diff > 0)
        # self.logger.debug('Binary Diff ' + str(diff_binary))
        # raise Exception('asdf')
        # By dividing by the correct max lengths of each row, we satisfy the above MUST CONDITION
        diff_res = np.sum(diff_binary, axis=-1) / ref_matrix_candidate_l
        # self.logger.debug('Measure result ' + str(diff_res))

        closest = np.argsort(diff_res)
        top_texts, top_dist = [ref_text_list[i] for i in closest], diff_res[closest].tolist()
        # self.logger.debug('Top texts ' + str(top_texts) + ', top dist ' + str(top_dist))

        return top_texts[0:min(top_k, len(top_texts))], top_dist[0:min(top_k, len(top_dist))]


class TextDiffCharDiffUnitTest:
    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        obj = TextDiffCharDiff(log_time_profilings=False, logger=self.logger)

        s = "privet medved"

        # Check chardiff model is correct
        AO = obj.APPEND_ORDINAL_DEFAULT
        s_rep = obj.get_text_model(
            text = s,
            model_params = {'ref_str_len': len(s)+2, 'append_ordinal': AO},
        )
        assert s_rep == [
            0x70, 0x72, 0x69, 0x76, 0x65, 0x74, 0x20, 0x6d, 0x65, 0x64, 0x76, 0x65, 0x64, AO, AO], \
            'Wrong chardiff model for "' + str(s) + '": ' + str(s_rep)

        #
        # Important test to ensure textdiff between 2 texts are the same regardless of reference fixed length
        #
        cand_txt = 'privet medved'
        # Check chardiff model again by different lengths
        for txt_compare, exp_d, ref_len in [
            # Should be no difference regardless of reference length 30 or more
            ('privet medved', 0., 30), ('privet medved', 0., 30+np.random.randint(low=50, high=100)),
            # Should be no difference regardless of reference length 30 or more
            ('privet med', 0.23076923076923078, 30),
            ('privet med', 0.23076923076923078, 30+np.random.randint(low=50, high=100)),
            # Should be no difference regardless of reference length 30 or more
            ('privet', 0.5384615384615384, 30),
            ('privet', 0.5384615384615384, 30+np.random.randint(low=50, high=100)),
            # Should be no difference regardless of reference length 30 or more
            ('', 1.0, 30), ('', 1.0, 30+np.random.randint(low=50, high=100)),
        ]:
            model_params = obj.get_model_params(
                ref_str_len = ref_len,
                append_ordinal = obj.APPEND_ORDINAL_DEFAULT,
            )
            top_texts, top_diffs = obj.text_difference(
                candidate_text = cand_txt,
                ref_text_list = [txt_compare],
                model_params = model_params,
            )
            self.logger.info(
                'chardiff for "' + str(cand_txt) + '" vs "' + str(txt_compare) + '", ref len ' + str(ref_len)
                + ': ' + str(list(zip(top_texts, top_diffs)))
            )
            assert top_diffs[0] == exp_d, \
                'chardiff for "' + str(cand_txt) + '" vs "' + str(txt_compare) + '", ref len ' + str(ref_len) \
                + ': ' + str(list(zip(top_texts, top_diffs))) + ' top diff ' + str(top_diffs[0]) + ' not ' + str(exp_d)

        # Check both metric is correct by top_k
        s_ref = 'hi how are u'
        test_text_list = [s_ref, 'how are you', 'how are', 'рш0рщц0фку0нщг']
        close_texts, close_scores = obj.text_difference(
            candidate_text = s_ref,
            ref_text_list = test_text_list,
            top_k = 4,
            model_params = obj.get_model_params(ref_str_len=50, append_ordinal=TextDiffCharDiff.APPEND_ORDINAL_DEFAULT),
        )
        self.logger.info(
            'Close texts and scores result: ' + str(list(zip(close_texts, close_scores)))
        )
        exp_scores = [0.0, 0.9166666666666666, 0.9166666666666666, 1.0]
        exp_texts = ['hi how are u', 'how are you', 'how are', 'рш0рщц0фку0нщг']
        assert close_texts == exp_texts, 'Close texts ' + str(close_texts) + ' not ' + str(exp_texts)
        assert close_scores == exp_scores,  'Close scores ' + str(close_scores) + ' not ' + str(exp_scores)

        self.logger.info('ALL TESTS PASSED')
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    TextDiffCharDiffUnitTest(logger=lgr).test()
    exit(0)
