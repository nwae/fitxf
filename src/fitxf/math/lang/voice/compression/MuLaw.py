import logging
import numpy as np
from fitxf.utils import Logging


# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
class Mulaw:

    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def u_law_enc(self, x, mu = 255):
        assert np.max(np.abs(x)) <= 1
        sgn = -1 * (x < 0) + 1 * (x >= 0)
        self.logger.debug(sgn)
        y = sgn * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
        return y

    def u_law_dec(self, y, mu = 255):
        sgn = -1 * (y < 0) + 1 * (y >= 0)
        x = sgn * ( (1 + mu) ** np.abs(y) - 1) / mu
        return x


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    ml = Mulaw(logger=lgr)
    x = 2 * ( (np.arange(101) / 100) - 0.5)
    lgr.info(x)
    x_enc = ml.u_law_enc(x=x)
    lgr.info(x_enc)
    exit(0)
