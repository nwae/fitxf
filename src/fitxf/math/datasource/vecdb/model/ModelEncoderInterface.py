import logging


class ModelEncoderInterface:
    def __init__(
            self,
            # can be LLM model files, or whatever
            cache_folder: str = None,
            logger = None,
    ):
        self.cache_folder = cache_folder
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def get_model_name(self):
        raise Exception('Must be implemented by derived class')

    def get_model_path(self):
        raise Exception('Must be implemented by derived class')

    def encode(self, content_list, return_tensors: str = 'np'):
        raise Exception('Must be implemented by derived class')
