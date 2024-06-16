from fitxf.math.lang.encode.LangModelInterface import LangModelInterface
from fitxf.math.utils.Singleton import Singleton


class LangModelPtSingleton:

    @staticmethod
    def get_singleton(
            LmClass,    # class type
            model_name = None,
            cache_folder = None,
            include_tokenizer = False,
            logger = None,
            return_key = False,
    ) -> LangModelInterface:
        # Don't include lang, because it may change to "multi"
        key_id = str(LmClass) + '.model_name=' + str(model_name) \
                 + '.cache_folder=' + str(cache_folder) + '.include_tokenizer=' + str(include_tokenizer)
        sgt = Singleton(
            class_type = LmClass,
            logger = logger,
        ).get_singleton(
            key_id,
            model_name,
            cache_folder,
            include_tokenizer,
            logger,
        )
        return (sgt, key_id) if return_key else sgt
