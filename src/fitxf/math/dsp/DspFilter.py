import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from fitxf.math.utils.Env import Env
from fitxf.math.utils.Logging import Logging


class DspFilter:

    def __init__(
            self,
            logger: Logging = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)
        return b, a

    def butter_bandpass_filter(self, data: np.ndarray, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

    def bandpass_filter(self, buffer: np.ndarray, lowcut, highcut, fs):
        return self.butter_bandpass_filter(data=buffer, lowcut=lowcut, highcut=highcut, fs=fs, order=6)


if __name__ == '__main__':
    er = Env()
    Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env')
    exit(0)
