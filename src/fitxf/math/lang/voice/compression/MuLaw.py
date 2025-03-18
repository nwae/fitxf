import logging
import numpy as np
from fitxf.utils import Logging


# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
class Mulaw:

    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def u_law_enc(self, x, mu = 255):
        assert abs(x) <= 1
        sgn = 1 if x >= 0 else -1
        y = sgn * np.log(1 + mu * abs(x)) / np.log(1 + mu)
        return y

    def u_law_dec(self, y, mu = 255):
        sgn = 1 if y >= 0 else -1
        x = sgn * ( (1 + mu) ** abs(y) - 1) / mu
        return x


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    ml = Mulaw(logger=lgr)
    exit(0)
