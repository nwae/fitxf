import logging
import pandas as pd
from collections import Counter, defaultdict
from fitxf.math.lang.tokenizer.TokenizerInf import TokenizerInterface
from fitxf.math.lang.tokenizer.TokenizerAuto import TokenizerAuto
from fitxf.utils import Logging, Env, Pandas


# Modified & optimized from:
#    https://github.com/DolbyUUU/byte_pair_encoding_BPE_subword_tokenization_implementation_python/blob/main/BPE.py
class TokenizerBpe(TokenizerInterface):
    """Byte-Pair Encoding: Subword-based tokenization algorithm."""

    def __init__(
            self,
            model_name: str,
            logger: logging.Logger | None = None,
    ):
        super().__init__(
            model_name = model_name,
            logger = logger,
        )

        # pre-tokenize the corpus into words, BERT pre-tokenizer is used here
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer_base = TokenizerAuto(
            model_name = self.model_name,
            logger = self.logger,
        )
        self.word_freqs = defaultdict(int)
        self.splits = {}
        # Final BPE "vocabulary" will be contained here in merges.
        # e.g. {("a", "n"): "an", ("an", "d"): "and", ("l", "a"): "la"..}
        self.merges = {}
        return

    def get_vocab_size(self) -> int:
        return len(self.merges)
        # return self.tokenizer_base.get_vocab_size()

    def get_cls_token(self) -> str: return self.tokenizer_base.get_cls_token()
    def get_cls_token_id(self) -> int: return self.tokenizer_base.get_cls_token_id()

    def get_sep_token(self) -> str: return self.tokenizer_base.get_sep_token()
    def get_sep_token_id(self) -> int: return self.tokenizer_base.get_sep_token_id()

    def get_pad_token(self) -> str: return self.tokenizer_base.get_pad_token()
    def get_pad_token_id(self) -> int: return self.tokenizer_base.get_pad_token_id()

    def get_unk_token(self) -> str: return self.tokenizer_base.get_unk_token()
    def get_unk_token_id(self) -> int: return self.tokenizer_base.get_unk_token_id()

    def get_mask_token(self) -> str: return self.tokenizer_base.get_mask_token()
    def get_mask_token_id(self) -> str: return self.tokenizer_base.get_mask_token_id()

    def get_special_tokens(self) -> dict:
        return self.tokenizer_base.get_special_tokens()

    def train(
            self,
            corpus: list,
            target_vocab_size: int = 0,
    ):
        base_vocab_size = self.tokenizer_base.get_vocab_size()

        # compute the frequencies of each word in the corpus
        for text in corpus:
            words_with_offsets = self.tokenizer_base.tokenize_into_words_and_offsets(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] += 1

        df_word = pd.DataFrame.from_records(data=[{'word': w, 'freq': f} for w, f in self.word_freqs.items()])
        df_word = df_word.sort_values(by=['freq', 'word'], ascending=False)
        df_word = df_word.reset_index(drop=True)
        self.logger.debug('Words freq: ' + str(df_word))

        # compute the base vocabulary of all characters in the corpus
        alphabet = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()
        self.logger.info('Alphabets: ' + str(alphabet) + ', total alphabets ' + str(len(alphabet)))

        if target_vocab_size <= 0:
            target_vocab_size = len(alphabet) * 10
            self.logger.info('Estimating target vocab size as ' + str(target_vocab_size))

        # add the special token </w> at the beginning of the vocabulary
        # vocab = ["</w>"] + alphabet.copy()
        vocab = alphabet.copy()
        self.logger.info('Initial vocab: ' + str(vocab) + ' (length ' + str(len(vocab)) + ')')

        # split each word into individual characters before training
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}
        self.logger.info(
            'Initial sample 10 splits: ' + str({k: v for i, (k, v) in enumerate(self.splits.items()) if i < 10})
            + ' of total splits ' + str(len(self.splits))
        )

        disallowed_specials_for_check_ids = set(self.tokenizer_base.get_special_tokens().keys())

        iteration = 0
        new_id = int(base_vocab_size)

        assert self.tokenizer_base.decode(token_ids=[new_id]) == "", \
            'Before train, the start new id ' + str(base_vocab_size) + ' must be unused'

        # merge the most frequent pair iteratively until the vocabulary size is reached
        while len(vocab) < target_vocab_size:
            iteration += 1
            # compute the frequency of each pair
            pair_freqs = self.compute_pair_freqs()

            # find the most frequent pair
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            if best_pair == "":
                break

            self.logger.info(
                'At iteration #' + str(iteration) + ', best pair "' + str(best_pair) + '" at frequency ' + str(max_freq)
            )

            # merge the most frequent pair
            self.splits = self.merge_pair(*best_pair)
            best_pair_merge_str = best_pair[0] + best_pair[1]

            # Check to see the decoded IDs
            ids_pair = self.tokenizer_base.tokenize(
                text = best_pair_merge_str,
                disallowed_special = disallowed_specials_for_check_ids,
            )
            is_new_pair = len(ids_pair) > 1
            if is_new_pair:
                pair_id = new_id
                new_id += 1
            else:
                pair_id = -1

            self.logger.info(
                'IDs for best pair ' + str(best_pair) + ' "' + str(best_pair_merge_str) + '": ' + str(ids_pair)
                + ' Is new pair ' + str(is_new_pair) + ', pair ID ' + str(pair_id)
            )
            if not is_new_pair:
                continue

            self.merges[best_pair] = {
                'pair_string': best_pair_merge_str,
                'iter': iteration,
                'freq': max_freq,
                'ids': ids_pair,
                'new_pair': is_new_pair,
                'pair_id': pair_id,
            }

            vocab.append(best_pair_merge_str)
        df_merges = pd.DataFrame.from_records(data=[
            {
                'chars': chars,
                'pair_string': d['pair_string'],
                'iter': d['iter'],
                'freq': d['freq'],
                'ids': d['ids'],
                'new_pair': d['new_pair'],
                'pair_id': d['pair_id'],
            } for chars, d in self.merges.items()
        ])
        df_merges = df_merges.sort_values(by='iter', ascending=True)
        df_merges = df_merges.reset_index(drop=True)
        # Vocab length = merges length + alphabets length
        assert len(df_merges) + len(alphabet) == len(vocab)
        self.logger.info('Final vocab length ' + str(len(vocab)) + ', merges ' + str(df_merges))
        return self.merges

    def compute_pair_freqs(self):
        """Compute the frequency of each pair."""

        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b):
        """Merge the given pair."""

        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits

    def tokenize(
            self,
            text: str,
            allowed_special: set = (),
            disallowed_special: set = (),
            # return_objects: list | tuple = ('id',),
    ) -> list:
        """Tokenize a given text with trained BPE tokenizer (including pre-tokenization, split, and merge)."""

        # TODO Shouldn't this step be same with when we trained it?
        # pre_tokenize_result = self.tokenizer_base.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenize_result = self.tokenizer_base.tokenize_into_words_and_offsets(text=text)
        self.logger.info('Pre tokenize result: ' + str(pre_tokenize_result))

        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits_text = [[l for l in word] for word in pre_tokenized_text]

        for pair, d_merge in self.merges.items():
            merge = d_merge['pair_string']
            for idx, split in enumerate(splits_text):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits_text[idx] = split
        result = sum(splits_text, [])
        return result

    def tokenize_into_words_and_offsets(
            self,
            text: str,
    ) -> list:
        return self.tokenizer_base.tokenize_into_words_and_offsets(text=text)

    # TODO not yet ready
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


class TokenizerBpeUnitTest:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        corpus = ['menya zovut ai', 'kak tebya zovut', 'kak on zovut', 'evo imya ia']
        self.logger.info('Corpus length ' + str(len(corpus)))

        tknzr = TokenizerBpe(
            model_name = TokenizerAuto.SUPPORTED_MODELS[0],
            logger = self.logger,
        )
        self.logger.info('Vocab size: ' + str(tknzr.get_vocab_size()))
        self.logger.info('Special tokens: ' + str(tknzr.get_special_tokens()))

        tknzr.train(
            corpus = corpus,
            target_vocab_size = 0,
        )

        for i, text in enumerate([
            "tiktoken is a fast and efficient tokenizer.",
            'on zovut imyanuel',
        ]):
            # Encode text into tokens
            toks = tknzr.tokenize(
                text = text,
                disallowed_special = set(tknzr.get_special_tokens().keys()),
            )
            toks_unicode = tknzr.tokenize_unicode_chars(text=text)
            words_offset = tknzr.tokenize_into_words_and_offsets(text=text)
            # The count of character per token can be < 1 because a token can be as small as a byte, whereas
            # a character can be 1 (ascii) to 4 (unicode) bytes
            char_per_tok = round(len(text) / len(toks), 2)
            unicode_per_tok = round(len(toks_unicode) / len(toks), 2)
            self.logger.info(
                '#' + str(i) + ' Tokens for "' + str(text) + '": ' + str(toks) + ' (unicode ' + str(toks_unicode)
                + ', words/offset ' + str(words_offset) + ')'
            )
            self.logger.info(
                '#' + str(i) + ' Word length ' + str(len(text.split(" "))) + ', char length ' + str(len(text))
                + ', token length ' + str(len(toks)) + ', avg char per token ' + str(char_per_tok)
                + ', unicode length to tokenized ratio ' + str(unicode_per_tok)
            )
            continue

        return


if __name__ == '__main__':
    Pandas.increase_display()
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)

    # get some sample data
    ev = Env(logger=lgr)

    TokenizerBpeUnitTest(logger=lgr).test()
    exit(0)

    tknzr = TokenizerBpe(
        model_name = TokenizerAuto.SUPPORTED_MODELS[0],
        logger = lgr,
    )

    for i, text in enumerate([
        "tiktoken is a fast and efficient tokenizer.",
        'Китай модернизирует армию с упором на кибервойну',
        'США вовлечены в космическую гонку с Китаем',
        'рассматривает Пекин как «угрозу растущего влияния»',
        '港澳台同胞和海外侨胞：祖国强大是我们的自豪',
        '"다탄두 각개 목표 설정 재돌입체"(MIRV)를 탑재 가능하다고 설명했다.',
        # Bengali (Sylheti dialect)
        'অইল ভাত',  # Cooked rice (পাক হয়েছে যে ভাত)
        'অকতে জরুর',  # Absolutely necessary (নিতান্ত  প্রয়োজনীয়)
        'আউয়া যাওয়া',  # Stupid, senseless (বোকা, অবোধ)
        # Check ids for specials
        '<|startoftext|><|endoftext|>',
    ]):
        # Encode text into tokens
        toks = tknzr.tokenize(
            text = text,
            disallowed_special = set(tknzr.get_special_tokens().keys()),
        )
        toks_unicode = tknzr.tokenize_unicode_chars(text=text)
        words_offset = tknzr.tokenize_into_words_and_offsets(text=text)
        # The count of character per token can be < 1 because a token can be as small as a byte, whereas
        # a character can be 1 (ascii) to 4 (unicode) bytes
        char_per_tok = round(len(text) / len(toks), 2)
        unicode_per_tok = round(len(toks_unicode) / len(toks), 2)
        lgr.info(
            '#' + str(i) + ' Tokens for "' + str(text) + '": ' + str(toks) + ' (unicode ' + str(toks_unicode)
            + ', words/offset ' + str(words_offset) + ')'
        )
        lgr.info(
            '#' + str(i) + ' Word length ' + str(len(text.split(" "))) + ', char length ' + str(len(text))
            + ', token length ' + str(len(toks)) + ', avg char per token ' + str(char_per_tok)
            + ', unicode length to tokenized ratio ' + str(unicode_per_tok)
        )

        # Decode tokens back into text
        decoded_text = tknzr.decode(token_ids=toks, include_special_tokens=False)
        lgr.info('#' + str(i) + ' Decode: ' + str(decoded_text))
        if text != decoded_text:
            lgr.error('Decoded text not ok:\n"' + str(decoded_text) + '" not:\n"' + str(text) + '"')
        else:
            lgr.error('Decoded text ok "' + str(decoded_text) + '" identical to original text "' + str(text) + '"')
    exit(0)
