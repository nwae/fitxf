import logging
import numpy as np
from nwae.math.utils.Env import Env
from nwae.math.utils.Singleton import Singleton
from nwae.math.utils.Profile import Profiling


class LangModelInterface:

    RANDOM_TEXT = \
"""2018년 10월 JYP엔터테인먼트와 정식 매니지먼트 계약을 맺었었다. 이후, 2019년 8월 7일 아티스트컴퍼니와 공식 매니지먼트 계약을 맺었다.
드라마 마녀의 법정, 배드파파, 나쁜 형사에 연달아 출연, 연기 스펙트럼을 쌓아 왔다. 특히, 영화 변신(2019)에서 사춘기 고등학생의 심리를
실감나게 표현해 일상 연기부터 섬뜩한 악마 연기까지 완벽 소화하여 호평받았다. 또한, 드라마 나의 나라(2019)에서는 서휘(양세종)의 동생으로
남선호(우도환)를 좋아하는 서연 역으로 출연하여 첫 사극 도전임에도 안정적인 캐릭터 소화력을 보여줘 ‘괴물 신인’으로 떠올랐다.
2020년, 2021년 tvN 드라마 슬기로운 의사생활에 장윤복 역으로 출연했다. 어리바리한 매력이 있는 쌍둥이 홍도와는 다르게 똘똘하고 열정 넘치는
의대생 윤복 역할을 맡았고, 이후 시즌 2에서는 의사 면허 합격 이후 의사 인턴 역할을 맡았다.
2021년 말부터 2022년 초까지 KBS2 드라마 학교 2021에서 진지원 역으로 출연했다.
2022년, 동명의 네이버 웹툰을 원작으로 한 넷플릭스 오리지널 드라마 지금 우리 학교는에서 반장 최남라 역으로 출연했다. 지금 우리 학교는을
통해 앞으로의 행보가 기대되는 배우로 성장하였다. 극중에서 직전작들[8]과 전혀 다른 캐릭터를 연기했는데,[9] 최남라를 잘 표현해내며 배우로서
성장 가능성을 시사하였다.
"""

    @staticmethod
    def get_singleton(
            LmClass,    # class type
            lang,
            model_name = None,
            cache_folder = None,
            include_tokenizer = False,
            params_other = None,
            logger = None,
            return_key = False,
    ):
        # Don't include lang, because it may change to "multi"
        key_id = str(LmClass) + '.model_name=' + str(model_name) \
                 + '.cache_folder=' + str(cache_folder) + '.include_tokenizer=' + str(include_tokenizer)
        sgt = Singleton(
            class_type = LmClass,
            logger = logger,
        ).get_singleton(
            key_id,
            lang,
            model_name,
            cache_folder,
            include_tokenizer,
            params_other,
            logger,
        )
        return (sgt, key_id) if return_key else sgt

    def __init__(
            self,
            lang,
            model_name = None,
            cache_folder = None,
            include_tokenizer = False,
            params_other = None,
            logger = None,
    ):
        self.lang = lang
        self.cache_folder = cache_folder if cache_folder is not None else Env.get_home_download_dir()
        self.model_name = model_name
        self.logger = logger if logger is not None else logging.getLogger()
        self.include_tokenizer = include_tokenizer
        self.params_other = params_other if params_other is not None else {}

        # local model path
        self.model_path = None
        return

    def get_name(self):
        return self.model_name

    def get_model_path(self):
        return self.model_path

    def encode(
            self,
            text_list,
            # max length has got no meaning
            maxlen = None,
            return_tensors = 'pt',
            # does not apply here since we can't see the tokenization
            return_attnmasks = False,
            params_other = None,
    ):
        raise Exception('Must be implemented in derived class')

    # Language models usually have max number of tokens, which depends on
    # the tokenization model, either WordPiece, BPE, etc.
    # Thus it is impossible to relate that to how many max characters.
    # It also depends on language, this is only for English.
    # Below algorithm is just a rough guess.
    # TODO Separate out random test text into .txt files for all common languages
    def self_discover_max_len(
            self,
            sample_text = RANDOM_TEXT,
            # last n characters just to avoid accidental errors,
            # make sure this number is larger than the longest word
            last_n_true = 50,
    ):
        n = 100
        last_n_true_count = 0
        emb_prev = None
        # When limit of max tokens reached, we assume the embedding no longer changes
        while n < len(sample_text):
            text = sample_text[0:n]
            self.logger.debug('n = ' + str(n) + ', text char len=' + str(n))
            emb, _ = self.encode(
                text_list = [text],
                return_tensors = 'np',
                return_attnmasks = True,
            )

            if emb_prev is not None:
                is_same = np.sum(emb - emb_prev)**2 < 0.000000001
                self.logger.debug('Is embedding same with previous = ' + str(is_same))
                if is_same:
                    last_n_true_count += 1
                    if last_n_true_count >= last_n_true:
                        self.logger.info(
                            'Embedding remained the same for the last ' + str(last_n_true)
                            + ' characters, starting from length ' + str(n-last_n_true) + '.'
                        )
                        return len(text)-last_n_true, text[0:n-last_n_true]
                else:
                    last_n_true_count = 0

            emb_prev = emb
            n += 1
        # result not found
        return len(sample_text), sample_text

    def speed_test(
            self,
            sentences: list,
            min_rounds = 1000,
    ):
        n = 0
        prof = Profiling(logger=self.logger)
        start_time = prof.start()
        self.logger.info('Start speed test for model "' + str(self.model_name) + '" time ' + str(start_time))
        while True:
            if n >= min_rounds:
                break
            for s in sentences:
                self.encode(text_list=[s])
                n += 1

        diftime = prof.stop() - start_time
        diftime_secs = diftime.days * 86400 + diftime.seconds + diftime.microseconds / 1000000
        rps = n / diftime_secs
        self.logger.info(
            'Model speed test "' + str(self.model_name) + '", rps ' + str(round(rps, 4)) + ', done ' + str(n)
            + ' in ' + str(diftime_secs) + 's'
        )
        return rps


if __name__ == '__main__':
    print(LangModelInterface.RANDOM_TEXT)
    exit(0)
