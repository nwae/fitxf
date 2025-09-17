import logging
from fitxf.math.utils.Logging import Logging


class TokenizerInterface:

    def __init__(
            self,
            model_name: str | None = None,
            logger: logging.Logger | None = None,
    ):
        self.model_name = model_name
        self.logger = logger if logger is not None else logging.getLogger()
        self.tokenizer = None
        return

    def get_vocab_size(self) -> int: raise Exception('Must be implemented in derived class')

    def get_cls_token(self) -> str: raise Exception('Must be implemented in derived class')
    def get_cls_token_id(self) -> int: raise Exception('Must be implemented in derived class')

    def get_sep_token(self) -> str: raise Exception('Must be implemented in derived class')
    def get_sep_token_id(self) -> int: raise Exception('Must be implemented in derived class')

    def get_pad_token(self) -> str: raise Exception('Must be implemented in derived class')
    def get_pad_token_id(self) -> int: raise Exception('Must be implemented in derived class')

    def get_unk_token(self) -> str: raise Exception('Must be implemented in derived class')
    def get_unk_token_id(self) -> int: raise Exception('Must be implemented in derived class')

    def get_mask_token(self) -> str: raise Exception('Must be implemented in derived class')
    def get_mask_token_id(self) -> int: raise Exception('Must be implemented in derived class')

    def get_wordsep_token(self) -> str: raise Exception('Must be implemented in derived class')
    def get_wordsep_token_id(self) -> int: raise Exception('Must be implemented in derived class')

    def get_special_tokens(self) -> dict:
        return {
            self.get_cls_token(): self.get_cls_token_id(),
            self.get_sep_token(): self.get_sep_token_id(),
            self.get_pad_token(): self.get_pad_token_id(),
            self.get_unk_token(): self.get_unk_token_id(),
            self.get_mask_token(): self.get_mask_token_id(),
        }

    def get_id_token_map(self):
        sz = self.get_vocab_size()
        map_id_tok = {}
        for i in range(sz):
            tok = self.decode(token_ids=[i])
            # Invalid id will be mapped to empty string usually
            if len(tok) == 0:
                self.logger.warning('Invalid id probably ' + str(i) + '. Id mapped to empty string')
            map_id_tok[i] = tok
            self.logger.debug("Map id " + str(i) + ' to "' + str(tok) + '"')
        return map_id_tok

    def tokenize(
            self,
            text: str,
            allowed_special: set = (),
            disallowed_special: set = (),
            # return_objects: list | tuple = ('id',),
    ) -> list:
        raise Exception('Must be implemented in derived class')

    def tokenize_into_words_and_offsets(
            self,
            text: str,
    ) -> list:
        raise Exception('Must be implemented in derived class')

    def tokenize_unicode_chars(
            self,
            text: str,
    ):
        # if {'token', 'id', 'offset'}.difference(set(return_objects)) == set():
        #     return [(c, ord(c), (i, i+1)) for i, c in enumerate(text)]
        # elif {'token', 'id', 'offset'}.difference(set(return_objects)) == {'offset'}:
        #     # only request for token & id
        #     return [(c, ord(c)) for _, c in enumerate(text)]
        # elif {'token', 'id', 'offset'}.difference(set(return_objects)) == {'token', 'offset'}:
        #     # only request for id
        return [ord(c) for _, c in enumerate(text)]
        # else:
        #     raise Exception('Not supported return objects: ' + str(return_objects))

    def decode(
            self,
            token_ids: list,
            include_special_tokens: bool = True,
    ) -> str:
        raise Exception('Must be implemented in derived class')

    def decode_unicode_tokens(
            self,
            tokens_ids: list,
    ):
        return ''.join([chr(t) for t in tokens_ids])


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)

    tknzr = TokenizerInterface(
        logger = lgr,
    )

    for i, text in enumerate([
        "tiktoken is a fast and efficient tokenizer.",
        'Китай модернизирует армию с упором на кибервойну',
        'США вовлечены в космическую гонку с Китаем',
        'рассматривает Пекин как «угрозу растущего влияния»',
        '港澳台同胞和海外侨胞：祖国强大是我们的自豪',
        '"다탄두 각개 목표 설정 재돌입체"(MIRV)를 탑재 가능하다고 설명했다.',
    ]):
        # Encode text into tokens
        toks = tknzr.tokenize_unicode_chars(text=text)
        # The count of character per token can be < 1 because a token can be as small as a byte, whereas
        # a character can be 1 (ascii) to 4 (unicode) bytes
        char_per_tok = round(len(text) / len(toks), 2)
        lgr.info('#' + str(i) + ' Tokens for "' + str(text) + '": ' + str(toks))
        lgr.info(
            '#' + str(i) + ' Word length ' + str(len(text.split(" "))) + ', char length ' + str(len(text))
            + ', token length ' + str(len(toks)) + ', avg char per token ' + str(char_per_tok)
        )

        # Decode tokens back into text
        decoded_text = tknzr.decode_unicode_tokens(tokens_ids=toks)
        lgr.info('#' + str(i) + ' Decode: ' + str(decoded_text))
        assert text == decoded_text
    exit(0)
