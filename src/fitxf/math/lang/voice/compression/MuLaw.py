import logging
import numpy as np
from fitxf.math.lang.voice.compression.MuLawBob404 import u_law_e, u_law_d
from fitxf.utils import Logging


# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
class Mulaw:

    BIN_INTERVALS = (
        # high, low, num sub intervals, code
        (8158, 4063, 16, 0x80),
        (4062, 2015, 16, 0x90),
        (2014,  991, 16, 0xA0),
        ( 990,  479, 16, 0xB0),
        ( 478,  223, 16, 0xC0),
        ( 222,   95, 16, 0xD0),
        (  94,   31, 16, 0xE0),
        (  30,    1, 15, 0xF0),
        (   0,    0,  1, 0xFF),
        (  -1,   -1,  1, 0x7F),
    )

    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def u_law_enc(self, x, mu = 255):
        assert np.max(np.abs(x)) <= 1
        sgn = -1 * (x < 0) + 1 * (x >= 0)
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
    x_dec = ml.u_law_dec(y=x_enc)
    tmp = np.array([x.tolist(), x_enc.tolist(), x_dec.tolist()]).transpose()
    lgr.info(tmp)
    exit(0)
