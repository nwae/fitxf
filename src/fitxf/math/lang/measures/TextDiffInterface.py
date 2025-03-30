import logging
from fitxf.utils import Profiling, Logging


#
#   Measure Consistency & Must Condition:
#     charfreq = 1.0 when no common character is shared (including space). e.g. 'IloveMcD' and 'ЯдюблюМзкД'
#     charfreq = 0.0 when count of characters are equal regardless of order
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
        raise Exception('Must be implemented by child class!!')


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    exit(0)
