import logging
from fitxf.math.lang.tokenizer.TokenizerInf import TokenizerInterface
from fitxf.math.utils.Logging import Logging


class TokenizerUnitTest:
    
    def __init__(
            self,
            tokenizer: TokenizerInterface,
            logger: logging.Logger | None = None,
    ):
        self.tokenizer = tokenizer
        self.logger = logger
        return

    def calculate_percent_similar(
            self,
            text_ori: str,
            text_decoded: str,
    ):
        len_ori = len(text_ori)
        cur_ori_idx = -1
        matches = []
        for i, c in enumerate(text_decoded):
            # copy to tmp value, so original variable won't be updated if character is not found at all
            tmp_idx = cur_ori_idx
            while tmp_idx + 1 < len_ori:
                tmp_idx += 1
                if c == text_ori[tmp_idx]:
                    self.logger.debug(
                        'Found character #' + str(i) + ' "' + str(c) + '" at original index ' + str(cur_ori_idx)
                        + ' "' + str(text_ori[:(tmp_idx+1)]) + '"'
                    )
                    matches.append(c)
                    # found something, update original variable
                    cur_ori_idx = tmp_idx
                    break
        return len(matches) / len_ori

    def test(
            self,
            test_langs: list | None = None,
            include_special_tokens: bool = False,
    ):
        self.logger.info('Vocab size: ' + str(self.tokenizer.get_vocab_size()))
        self.logger.info('Special tokens: ' + str(self.tokenizer.get_special_tokens()))

        all_special_tokens = self.tokenizer.get_special_tokens().keys()
        all_special_tokens_ids = self.tokenizer.get_special_tokens().values()
        cls, wordsep = self.tokenizer.get_cls_token(), self.tokenizer.get_wordsep_token()
        disallowed_specials = () if include_special_tokens else set(all_special_tokens)
        self.logger.info(
            'Disallowed specials: ' + str(disallowed_specials) + ', all special tokens ' + str(all_special_tokens)
            + ', word separator token "' + str(wordsep) + '"'
        )

        for i, (lang, text, _) in enumerate([
            ("en", "tiktoken is a fast and efficient tokenizer.", None),
            ("en", "several command-line tools and graphical utilities are available", None),
            ("ru", 'Китай модернизирует армию с упором на кибервойну', None),
            ("ru", 'США вовлечены в космическую гонку с Китаем', None),
            ("ru", 'рассматривает Пекин как «угрозу растущего влияния»', None),
            ("zh", '港澳台同胞和海外侨胞：祖国强大是我们的自豪', '港澳台同胞和海外侨胞:祖国强大是我们的自豪'),
            ("zh", '"다탄두 각개 목표 설정 재돌입체"(MIRV)를 탑재 가능하다고 설명했다.', None),
        ]):
            if test_langs:
                if lang not in test_langs:
                    self.logger.info('Ignore lang "' + str(lang) + '"')
                    continue

            # Encode text into tokens
            tok_ids = self.tokenizer.tokenize(
                text = text,
                disallowed_special = disallowed_specials,
            )
            toks_unicode = self.tokenizer.tokenize_unicode_chars(text=text)
            words_offset = self.tokenizer.tokenize_into_words_and_offsets(text=text)
            # The count of character per token can be < 1 because a token can be as small as a byte, whereas
            # a character can be 1 (ascii) to 4 (unicode) bytes
            char_per_tok = round(len(text) / len(tok_ids), 2)
            unicode_per_tok = round(len(toks_unicode) / len(tok_ids), 2)
            self.logger.info(
                '#' + str(i) + ' Tokens for "' + str(text) + '": ' + str(tok_ids) + ' (unicode ' + str(toks_unicode)
                + ', words/offset ' + str(words_offset) + ')'
            )
            self.logger.info(
                '#' + str(i) + ' Word length ' + str(len(text.split(" "))) + ', char length ' + str(len(text))
                + ', token length ' + str(len(tok_ids)) + ', avg char per token ' + str(char_per_tok)
                + ', unicode length to tokenized ratio ' + str(unicode_per_tok)
            )

            # decoded_text_expected = text if decoded_text_expected is None else decoded_text_expected
            # Decode tokens back into text
            decoded_text = self.tokenizer.decode(
                token_ids = tok_ids,
                include_special_tokens = include_special_tokens,
            )
            if include_special_tokens:
                text_compare = cls + wordsep + text + wordsep + cls
            else:
                text_compare = text
            self.logger.info('Comparing text "' + str(text_compare) + '" with decoded text "' + str(decoded_text) + '"')
            similarity = max(
                self.calculate_percent_similar(text_ori=text_compare, text_decoded=decoded_text),
                self.calculate_percent_similar(text_ori=decoded_text, text_decoded=text_compare),
            )
            self.logger.info('#' + str(i) + ' Similarity ' + str(similarity) + ', decoded text: ' + str(decoded_text))
            if text != decoded_text:
                self.logger.error('Decoded text not ok:\n"' + str(decoded_text) + '" not:\n"' + str(text_compare) + '"')
            else:
                self.logger.info('Decoded text ok "' + str(decoded_text) + '" identical to "' + str(text_compare) + '"')
            assert similarity > 0.88, \
                'Decoded text similarity ' + str(similarity) + ', unicode:\n' + str([ord(c) for c in decoded_text]) \
                + ' not:\n' + str([ord(c) for c in text])

            #
            # check if words are tokenized into the same ids
            # means it is OK to accumulate the token ids of tokenized pre-tokenization words:
            #    i.e. ids = tok(w1) + tok(w2) +..
            # however it is NOT OK (in general) to join the pre-tokenization words, then encode into IDs:
            #    i.e. ids != tok(w1 + w2 + ...)
            #
            ids_accum = []
            for w, _ in words_offset:
                ids_word = self.tokenizer.tokenize(
                    text = w,
                    disallowed_special = disallowed_specials,
                )
                self.logger.info('Word "' + str(w) + '" tokenized into ' + str(ids_word))
                ids_accum = ids_accum + ids_word
            # remove all special symbols not at the start/end
            ids_accum = [
                __id for __i, __id in enumerate(ids_accum)
                if (__id not in all_special_tokens_ids)
                   or ( (__id in all_special_tokens_ids) and (__i in [0, len(ids_accum)-1]) )
            ]
            self.logger.info("Accumulated tokens ids: " + str(ids_accum) + ', original token ids: ' + str(tok_ids))
            assert ids_accum == tok_ids, "Accumulated tokens ids:\n" + str(ids_accum) + ' not:\n' + str(tok_ids)

            #
            # It is generally NOT OK to join the pre-tokenization words, then encode into IDs:
            #    i.e. ids != tok(w1 + w2 + ...)
            # However they can be the same if we use the correct word separator to join the words.
            #
            pretok_words_joined = wordsep.join([w for w, _ in words_offset])
            pretok_tok_ids = self.tokenizer.tokenize(
                text = pretok_words_joined,
                disallowed_special = disallowed_specials,
            )
            is_same = pretok_tok_ids == tok_ids
            self.logger.info(
                'This model "' + str(self.tokenizer.model_name) + '" is consistent = ' + str(is_same)
                + '. IDs of joined words "' + str(pretok_words_joined) + '" ' + str(pretok_tok_ids)
                + ' ids of original text "' + str(text) + '" ' + str(tok_ids)
            )
            assert is_same
        self.logger.info(
            'TESTS PASSED (model "' + str(self.tokenizer.model_name) + '", include special '
            + str(include_special_tokens) + ')'
        )
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)

    tknzr = TokenizerInterface(
        logger = lgr,
    )
    # lgr.info('Vocab size: ' + str(tknzr.get_vocab_size()))
    # lgr.info('Special tokens: ' + str(tknzr.get_special_tokens()))

    tok_ut = TokenizerUnitTest(
        tokenizer = tknzr,
        logger = lgr,
    )
    tok_ut.test(test_langs=None)
    exit(0)
