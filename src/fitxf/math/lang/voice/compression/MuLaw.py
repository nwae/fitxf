import logging
import numpy as np
import pandas as pd
from fitxf.utils import Logging


raise Exception('Not yet supported')

# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
class Mulaw:

    MAX_POS = 0x1fde # 8158

    # high/low, low/high, num sub intervals, interval, code
    #    low[i] = low[i-1] + 2**(i+3)   for i >= 3
    # Thus
    #    low[i] = low[2] + 2**6 + 2**7 + ... + 2**(i+3)
    #           = low[2] + 64 (1 + 2 + .. + 2**(i-3))
    #           = 31 + 64 (2^(i-2) - 1)
    # Inversion to get i
    #    i = Floor( 2 + log2( 1 + ( low[i] - 31 ) / 64 ) )
    BIN_LOW_EDGES_INCLUSIVE = (     0,      1,     31,     95,    223,    479,    991,   2015,  4063,  8159,)
    BIN_CODES =               ('0xff', '0xf0', '0xe0', '0xd0', '0xc0', '0xb0', '0xa0', '0x90', '0x80', '-',)

    # only need positive bins, negative bins are just bit inverse of the positive side
    BIN_INTERVALS = (
        ( 8158, 4063, 16, 256, 0x80), # i=8
        ( 4062, 2015, 16, 128, 0x90), # i=7
        ( 2014,  991, 16,  64, 0xA0), # i=6
        (  990,  479, 16,  32, 0xB0), # i=5
        (  478,  223, 16,  16, 0xC0), # i=4
        (  222,   95, 16,   8, 0xD0), # i=3
        (   94,   31, 16,   4, 0xE0), # i=2
        (   30,    1, 15,   2, 0xF0), # i=1
        (    0,    0,  1,   0, 0xFF), # i=0
        # The lower edge is just the inverted bits of the positive values on top
        # e.g. ~8158 = -8159, ~4062 = -4063, ~2014 = -2015, ~990 = -991,
        #      ~478 = -479, ~222 = -223, ~94 = -95, ~30 = -31, ~0 = -1
        # As for the code numbers, they are the bit AND with 0x7F, or zeroing the 8th bit from the right
        # e.g. hex(0x80 & 0x7F) = 0x10
        #      hex(0x90 & 0x7F) = 0x20
        #      ...
        #      hex(0xFF & 0x7F) = 0x7F
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

        low = np.array(self.BIN_LOW_EDGES_INCLUSIVE)
        check_edges = [0, 1, 31] + [low[i - 1] + 2 ** (i + 3) for i in range(3, 10, 1)]
        self.logger.info('Check edges ' + str(check_edges))
        assert list(self.BIN_LOW_EDGES_INCLUSIVE) == check_edges, \
            'Check edges ' + str(check_edges) + ' not ' + str(self.BIN_LOW_EDGES_INCLUSIVE)

        # check_bins = [0, 1, 2] + [2 + np.log2(1 + (low[i] - 31) / 64) for i in range(3, 10, 1)]
        check_bins = self.calculate_lower_edge_bin(value=low, sign=None).tolist()
        self.logger.info('Check bins ' + str(check_bins))
        assert np.arange(len(check_bins)).tolist() == check_bins, \
            'Check bins ' + str(check_bins) + ' not ' + str(np.arange(len(check_bins)))

        codes = np.array(self.BIN_CODES)
        check_codes = codes[np.array([int(bin) for bin in check_bins])]
        self.logger.info('Check codes ' + str(check_codes))
        assert list(self.BIN_CODES) == check_codes.tolist(), \
            'Check codes ' + str(check_codes) + ' not ' + str(self.BIN_CODES)
        return

    # interval can be scalar or numpy ndarray
    # def calculate_bin_interval(
    #         self,
    #         # int or numpy ndarray of integers
    #         interval,
    # ):
    #     # 31 will be at index 2
    #     return (1 * (interval == 1)) * 1 \
    #         + (1 * (interval == 2)) * 31 \
    #         + (1 * (interval > 2)) * ( 31 + 64 * ((2 ** (interval - 2)) - 1) )

    # value can be scalar or numpy ndarray
    def calculate_lower_edge_bin(
            self,
            value: np.ndarray,  # all non-negative values
            sign: np.ndarray,   # +1 or -1
    ):
        # 0 will be at index 0, 31 will be at index 2
        bins_lower_edge = (1 * (value == 0)) * np.array([0]).astype(np.int16) \
                          + (1 * (value == 1)) * np.array([1]).astype(np.int16) \
                          + (1 * (value > 1) & (value <=31)) * np.array([2]).astype(np.int16) \
                          + (1 * (value > 31)) * np.floor(2 + np.log2( 1 + ( value - 31 ) / 64 )).astype(np.int16)
        return bins_lower_edge

    def create_bins(self):
        # only need positive bins, negative bins are just bit inverse of the positive side
        self.edge_bins = np.array([v[1] for v in self.BIN_INTERVALS if v[1]>=0])[::-1]
        self.edge_bins_extended = np.array(self.edge_bins.tolist() + [self.MAX_VAL])

        self.num_interval_bins = np.array([v[2] for v in self.BIN_INTERVALS if v[1]>=0])[::-1]
        self.interval_bins = np.array([v[3] for v in self.BIN_INTERVALS if v[1]>=0])[::-1]
        self.code_bins = np.array([v[4] for v in self.BIN_INTERVALS if v[1]>=0])[::-1]
        self.logger.info('Code bins: ' + str([hex(x) for x in self.code_bins]))
        df = pd.DataFrame({
            'edge': self.edge_bins,
            'num_interval': self.num_interval_bins,
            'interval': self.interval_bins,
            'code': self.code_bins,
        })
        self.logger.info(df)
        self.logger.info('Edges in hex: ' + str([hex(v) for v in self.edge_bins]))
        self.logger.info('Bit Inversion of edges: ' + str([hex(~v) for v in self.edge_bins]))
        self.logger.info([self.edge_bins[i] - self.edge_bins[i-1] for i in range(len(self.edge_bins)) if i>0])
        return

    def u_law_enc(self, x, mu = 255):
        assert np.max(np.abs(x)) <= 1
        sgn = -1 * (x < 0) + 1 * (x >= 0)
        y = sgn * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

        y_pos = np.round(y * sgn * self.MAX_POS, decimals=0).astype(np.int16)
        self.logger.info('y_pos ' + str(y_pos))
        # Use inverse formula
        bin_lower = self.calculate_lower_edge_bin(value=y_pos, sign=sgn)
        bin_higher = bin_lower + 1
        self.logger.info('Bins for the values: ' + str(list(zip(bin_lower.tolist(), bin_higher.tolist()))))
        codes = self.code_bins[bin_lower]
        edges_lower = self.edge_bins[bin_lower]
        edges_higher = self.edge_bins[bin_higher]
        self.logger.info(codes)
        edges_lower = (-1 * (sgn == -1)) * (edges + 1) + (1 * (sgn == 1)) * edges
        codes = (1 * (sgn == -1)) * np.array([v & 0x7F for v in codes]) + (1 * (sgn == 1)) * codes
        self.logger.info('Value/Edges: ' + str(list(zip((y_pos*sgn).tolist(), edges.tolist()))))
        self.logger.info('Value/Codes: ' + str(list(zip((y_pos*sgn).tolist(), [hex(x) for x in codes.tolist()]))))
        raise Exception(list(zip(y_pos.tolist(), bin.tolist())))
        return y

    def u_law_dec(self, y, mu = 255):
        assert np.max(np.abs(y)) <= 1
        sgn = -1 * (y < 0) + 1 * (y >= 0)
        x = sgn * ( (1 + mu) ** np.abs(y) - 1) / mu

        m = self.edge_bins[0] * (x < 0) + self.edge_bins[-1] * (x >= 0)
        x_d = np.round(x * m, decimals=0)
        return x


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    ml = Mulaw(logger=lgr)
    # intervals = ml.calculate_bin_interval(interval=np.arange(8))
    # lgr.info('Intervals calculated: ' + str(intervals))
    value = np.array([0, 1, 30, 31, 94, 95, 222, 223, 478, 479, 990, 991, 2015, 4063, 8159])
    inv = ml.calculate_lower_edge_bin(
        value = value,
        sign = np.ones(len(value))
    )
    lgr.info('Inverse: ' + str(inv) + ', associated values ' + str(ml.edge_bins_extended[inv]))
    # raise Exception('asdf')
    x = 2 * ( (np.arange(101) / 100) - 0.5)
    lgr.info(x)
    x_enc = ml.u_law_enc(x=x)
    x_dec = ml.u_law_dec(y=x_enc)
    tmp = np.array([
        x.tolist(), x_enc.tolist(), x_dec.tolist()
    ]).transpose()
    lgr.info(tmp)
    exit(0)
