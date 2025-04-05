import logging
import numpy as np
import pandas as pd
from fitxf.utils import Logging


# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
class Mulaw:

    BIN_INTERVALS = (
        # high/low, low/high, num sub intervals, interval, code
        ( 8158, 4063, 16, 256, 0x80),
        ( 4062, 2015, 16, 128, 0x90),
        ( 2014,  991, 16,  64, 0xA0),
        (  990,  479, 16,  32, 0xB0),
        (  478,  223, 16,  16, 0xC0),
        (  222,   95, 16,   8, 0xD0),
        (   94,   31, 16,   4, 0xE0),
        (   30,    1, 15,   2, 0xF0),
        (    0,    0,  1,   0, 0xFF),
        (   -1,   -1,  1,   0, 0x7F),
        (  -31,   -2, 15,   2, 0x70),
        (  -95,  -32, 16,   4, 0x60),
        ( -223,  -96, 16,   8, 0x50),
        ( -479, -224, 16,  16, 0x40),
        ( -991, -480, 16,  32, 0x30),
        (-2015, -992, 16,  64, 0x20),
        (-4063,-2016, 16, 128, 0x10),
        (-8159,-4064, 16, 256, 0x00),
    )
    MAX_VAL = 8158

    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        self.create_bins()
        return

    def create_bins(self):
        self.edge_bins = np.array([v[0] for v in self.BIN_INTERVALS])[::-1]
        self.num_interval_bins = np.array([v[2] for v in self.BIN_INTERVALS])[::-1]
        self.interval_bins = np.array([v[3] for v in self.BIN_INTERVALS])[::-1]
        self.code_bins = np.array([v[4] for v in self.BIN_INTERVALS])[::-1]
        df = pd.DataFrame({
            'edge': self.edge_bins,
            'num_interval': self.num_interval_bins,
            'interval': self.interval_bins,
            'code': self.code_bins,
        })
        self.logger.info(df)
        return

    def u_law_enc_val(self, x, mu = 255):
        assert np.max(np.abs(x)) <= 1
        sgn = -1 * (x < 0) + 1 * (x >= 0)
        y = sgn * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
        return y

    def u_law_dec_val(self, y, mu = 255):
        assert np.max(np.abs(y)) <= 1
        sgn = -1 * (y < 0) + 1 * (y >= 0)
        x = sgn * ( (1 + mu) ** np.abs(y) - 1) / mu
        return x


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    ml = Mulaw(logger=lgr)
    x = 2 * ( (np.arange(101) / 100) - 0.5)
    lgr.info(x)
    x_enc = ml.u_law_enc_val(x=x)
    x_dec = ml.u_law_dec_val(y=x_enc)
    tmp = np.array([
        x.tolist(), x_enc.tolist(), x_dec.tolist()
    ]).transpose()
    lgr.info(tmp)
    exit(0)
