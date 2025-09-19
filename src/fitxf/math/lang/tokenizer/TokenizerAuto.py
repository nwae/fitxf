import logging
import torch
import numpy as np
from fitxf.math.lang.tokenizer.TokenizerInf import TokenizerInterface
from transformers import AutoTokenizer
from fitxf.math.utils.Logging import Logging


# See https://huggingface.co/docs/transformers/main/fast_tokenizers
class TokenizerAuto(TokenizerInterface):

    DEMO_MODELS = (
        'bert-base-uncased', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    )

    def __init__(
            self,
            model_name_or_path: str | None = "bert-base-uncased",
            logger: logging.Logger | None = None,
    ):
        super().__init__(
            model_name_or_path = model_name_or_path,
            logger = logger,
        )
        # Load an encoding for a specific model
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = self.model_name_or_path,
        )
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

    def get_wordsep_token(self) -> str:
        if self.model_name_or_path in ['bert-base-uncased']:
            return " "
        else:
            return ""

    def tokenize(
            self,
            text: str,
            allowed_special: set = (),
            disallowed_special: set = (),
    ) -> list:
        # tokens_and_idx_start_end = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        return self.tokenizer.tokenize(text)

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
        # self.logger.debug('Disallowed special ids: ' + str(disallowed_special_ids))

        if disallowed_special:
            token_ids = [tok for tok in ids if tok not in disallowed_special_ids]
        else:
            token_ids = ids
        # self.logger.debug('Filtered tokenization from:\n' + str(ids) + '\nto:\n' + str(token_ids))

        return token_ids

    def encode(
            self,
            text: str,
            allowed_special: set = (),
            disallowed_special: set = (),
            return_len: int = 0,
            # allowed values 'pt', 'np'
            return_tensor: str | None = None
    ) -> list | np.ndarray | torch.Tensor:
        # tokens_and_idx_start_end = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        ids = self.tokenizer.encode(text)
        ids_filtered = self.__filter_disallowed_ids(
            ids = ids,
            disallowed_special = disallowed_special,
        )
        if return_len > 0:
            if len(ids_filtered) >= return_len:
                ids_filtered = ids_filtered[:return_len]
            else:
                l_pad = return_len - len(ids_filtered)
                ids_filtered = ids_filtered + l_pad * [self.get_pad_token_id()]

        if return_tensor == 'pt':
            return torch.LongTensor(ids_filtered)
        elif return_tensor == 'np':
            return np.array(ids_filtered, dtype=np.int64)
        else:
            return ids_filtered

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

    def train(
            self,
            text_corpus: list,
            batch_size: int = 64,
            vocab_size: int | None = None,
            save_path: str | None = None,
    ):
        train_iter = self.get_training_corpus(
            text_list = text_corpus,
            batch_size = batch_size,
        )
        vocab_sz = vocab_size if vocab_size is not None else self.get_vocab_size()
        #
        # This will train a totally new tokenizer and forget everything in old tokenizer!
        #
        tokenizer_new = self.tokenizer.train_new_from_iterator(
            text_iterator = train_iter,
            vocab_size = vocab_sz,
        )
        self.logger.info('Successfully trained tokenizer')
        if save_path is not None:
            tokenizer_new.save_pretrained(save_path)
            self.logger.info('Trained tokenizer saved to "' + str(save_path) + '"')
        self.tokenizer = tokenizer_new
        return tokenizer_new


if __name__ == '__main__':
    from fitxf.math.lang.tokenizer.TokenizerUnitTest import TokenizerUnitTest
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)

    for model, langs_to_test, include_special_toks in [
        (TokenizerAuto.DEMO_MODELS[0], ['en', 'ru',], False),
        (TokenizerAuto.DEMO_MODELS[0], ['en', 'ru',], True),
        (TokenizerAuto.DEMO_MODELS[1], ['en', 'ru', 'zh',], False),
        (TokenizerAuto.DEMO_MODELS[1], ['en', 'ru', 'zh', ], True),
    ]:
        tknzr = TokenizerAuto(
            model_name_or_path = model,
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
        tok_ut.test(
            test_langs = langs_to_test,
            include_special_tokens = include_special_toks,
        )
    exit(0)
