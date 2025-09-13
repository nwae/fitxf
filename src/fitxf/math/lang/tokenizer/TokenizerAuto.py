import logging
from fitxf.math.lang.tokenizer.TokenizerInf import TokenizerInterface
from transformers import AutoTokenizer
from fitxf.math.utils.Logging import Logging


# See https://huggingface.co/docs/transformers/main/fast_tokenizers
class TokenizerAuto(TokenizerInterface):

    SUPPORTED_MODELS = (
        'bert-base-uncased', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    )

    def __init__(
            self,
            model_name: str | None = "bert-base-uncased",
            logger: logging.Logger | None = None,
    ):
        super().__init__(
            model_name = model_name,
            logger = logger,
        )
        if self.model_name not in self.SUPPORTED_MODELS:
            self.logger.warning('Unsupported (untested) model for tokenizer "' + str(self.model_name) + '"')
        # Load an encoding for a specific model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.specials_token_to_id = self.get_special_tokens()
        return

    def get_vocab_size(self) -> int: return self.tokenizer.vocab_size

    def get_cls_token(self) -> str: return self.tokenizer.cls_token
    def get_cls_token_id(self) -> int: return self.tokenizer.cls_token_id

    def get_sep_token(self) -> str: return self.tokenizer.sep_token
    def get_sep_token_id(self) -> int: return self.tokenizer.sep_token_id

    def get_pad_token(self) -> str: return self.tokenizer.pad_token
    def get_pad_token_id(self) -> int: return self.tokenizer.pad_token_id

    def get_unk_token(self) -> str: return self.tokenizer.unk_token
    def get_unk_token_id(self) -> int: return self.tokenizer.unk_token_id

    def get_mask_token(self) -> str: return self.tokenizer.mask_token
    def get_mask_token_id(self) -> str: return self.tokenizer.mask_token_id

    def tokenize(
            self,
            text: str,
            allowed_special: set = (),
            disallowed_special: set = (),
    ) -> list:
        # tokens_and_idx_start_end = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        ids = self.tokenizer.encode(text)
        return self.__filter_disallowed_ids(
            ids = ids,
            disallowed_special = disallowed_special,
        )

    def tokenize_into_words_and_offsets(
            self,
            text: str,
    ) -> list:
        tokens_and_idx_start_end = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        return tokens_and_idx_start_end

    def __filter_disallowed_ids(
            self,
            ids: list,
            disallowed_special: set = (),
    ) -> list:
        disallowed_special_ids = [self.specials_token_to_id[tok] for tok in disallowed_special]
        self.logger.debug('Disallowed special ids: ' + str(disallowed_special_ids))

        if disallowed_special:
            token_ids = [tok for tok in ids if tok not in disallowed_special_ids]
        else:
            token_ids = ids

        return token_ids

    def decode(
            self,
            token_ids: list,
            include_special_tokens: bool = True,
    ) -> str:
        if not include_special_tokens:
            d_specials = {id: tok for tok, id in self.get_special_tokens().items()}
            token_ids_tmp = [id for id in token_ids if id not in d_specials.keys()]
        else:
            token_ids_tmp = token_ids
        # Decode tokens back into text
        decoded_text = self.tokenizer.decode(token_ids_tmp)
        # Invalid id will be mapped to empty string
        if len(decoded_text) == 0:
            self.logger.warning('Invalid ids probably ' + str(token_ids_tmp) + '. Id mapped to empty string')
        return decoded_text


if __name__ == '__main__':
    from fitxf.math.lang.tokenizer.TokenizerUnitTest import TokenizerUnitTest
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)

    for model, langs_to_test in [
        (TokenizerAuto.SUPPORTED_MODELS[0], ['en', 'ru',],),
        (TokenizerAuto.SUPPORTED_MODELS[1], ['en', 'ru', 'zh',],),
    ]:
        tknzr = TokenizerAuto(
            model_name = model,
            logger = lgr,
        )
        lgr.info('Vocab size: ' + str(tknzr.get_vocab_size()))
        lgr.info('Special tokens: ' + str(tknzr.get_special_tokens()))

        # id_tok = tknzr.get_id_token_map()
        # lgr.info(id_tok)
        # invalid_tok = tknzr.decode(token_ids=[tknzr.get_vocab_size()])
        # lgr.info('oor: "' + str(invalid_tok) + '", length ' + str(len(invalid_tok)))

        tok_ut = TokenizerUnitTest(
            tokenizer = tknzr,
            logger = lgr,
        )
        tok_ut.test(test_langs=langs_to_test)
    exit(0)
