import logging
from fitxf.utils import Profiling, Logging


#
# Simple text models (no big data LLMs) for fast & convenient comparisons of text differences.
# Mainly for use in simple applications, requiring fast, simple metrics.
# The function textdiff(t1, t2):
#   Measure Consistency & Must Condition:
#     textdiff = 1.0 when totally "different" (depending on model definition) e.g. 'IloveMcD' and 'ЯдюблюМзкД'
#     textdiff = 0.0 when t1 == t2
#
class TextDiffInterface:

    def __init__(
            self,
            log_time_profilings = False,
            logger = None,
    ):
        self.log_time_profilings = log_time_profilings
        self.logger = logger if logger is not None else logging.getLogger()
        self.profiler = Profiling(logger=self.logger)
        return

    def get_model_params(self, **kwargs) -> dict:
        return {k: v for k, v in kwargs.items()}

    def get_text_model(
            self,
            text: str,
            model_params = {},
    ):
        raise Exception('Must be implemented by child class!!')

    def text_difference(
            self,
            candidate_text,
            ref_text_list,
            # option for user to pre-calculate to whichever text model being used
            candidate_text_model = None,
            ref_text_model_list = None,
            model_params = {},
            top_k = 3,
    ) -> tuple: # returns tuple of top text list & top scores list
        raise Exception('Must be implemented by child class!!')

    def text_similarity(
            self,
            candidate_text,
            ref_text_list,
            # option for user to pre-calculate to whichever text model being used
            candidate_text_model = None,
            ref_text_model_list = None,
            model_params = {},
            top_k = 3,
    ) -> tuple: # returns tuple of top text list & top scores list
        top_texts, top_scores = self.text_difference(
            candidate_text = candidate_text,
            ref_text_list = ref_text_list,
            candidate_text_model = candidate_text_model,
            ref_text_model_list = ref_text_model_list,
            model_params = model_params,
            top_k = top_k,
        )
        return [t for t in reversed(top_texts)], [v for v in reversed(top_scores)]


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    exit(0)
