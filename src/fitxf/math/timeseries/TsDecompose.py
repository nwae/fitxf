import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fitxf.math.dsp.Dft import Dft
# from statsmodels.tsa.seasonal import seasonal_decompose
from fitxf.utils import Logging


#
# https://en.wikipedia.org/wiki/Decomposition_of_time_series
#
class TsDecompose:

    def __init__(
            self,
            logger: Logging = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    #
    # Moving average
    #
    def calculate_ma(
            self,
            series: np.ndarray,
            weights: np.ndarray,
            method: str = 'np',
    ) -> np.ndarray:
        assert series.ndim == 1
        if method == 'np':
            # numpy convolve will extend the length of the original series, so we clip it
            return np.convolve(a=series, v=weights)[:len(series)]
        else:
            inv_w = np.flip(weights, axis=0)
            self.logger.info('Flipped weights ' + str(inv_w))
            l_add = len(weights) - 1
            series_extended = np.array([0.0]*l_add + series.tolist())
            self.logger.info('Series extended ' + str(series_extended))
            return np.array([np.sum(inv_w * series_extended[i:(i + l_add + 1)]) for i in range(len(series))])



class TsDecomposeUnitTest:
    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        ts_dec = TsDecompose(logger=self.logger)
        N = 10
        series = np.flip(np.arange(N) + 1, axis=0).astype(np.float32)
        weights = np.array([0.5, 0.3, 0.2])
        exp_ma = np.array([5.,  7.5, 8.7, 7.7, 6.7, 5.7, 4.7, 3.7, 2.7, 1.7])
        ma_1 = ts_dec.calculate_ma(series=series, weights=weights, method='manual')
        ma_2 = ts_dec.calculate_ma(series=series, weights=weights, method='np')
        self.logger.info('MA manual: ' + str(ma_1) + '\n, via numpy: ' + str(ma_2))
        assert np.sum((ma_1**2) - (ma_2**2)) < 0.0000000001, 'MA manual ' + str(ma_1) + ' not ' + str(ma_2)
        raise Exception('asdf')

        # Generate random time series, with cycle of sine
        N = 100
        k = 3
        t = np.arange(N).astype(np.float32)
        # random values from 0-10, add 2 cycles of sine pertubation
        y = np.sin(t * 2 * np.pi * k / N) + np.random.rand(N)
        self.logger.info('Generated time series length ' + str(len(y)) + ': ' + str(y))
        # plt.plot(t, y, marker='o', linestyle='-', color='b', label='Line 1')
        # plt.show()

        #
        # Do some statistical study
        #
        avg, var = np.mean(y), np.var(y)
        self.logger.info('Mean & var ' + str(avg) + ', ' + str(var))

        #
        # Calculate seasonality (if any) by DFT
        #
        # df = pd.DataFrame({'t': t, 'series': y})
        # df.reset_index(inplace=True)
        # df.set_index('t', inplace=True)
        # res = seasonal_decompose(x=df['series'], model='additive')
        # res.plot()
        # plt.show()
        dft_helper = Dft(logger=self.logger)
        dft = dft_helper.DFT(x=y)
        dft_mag = np.absolute(dft)
        self.logger.info('DFT (' + str(len(dft_mag)) + '): ' + str(dft_mag))
        plt.plot(t, dft_mag, marker='o', linestyle='-', color='b', label='DFT')
        plt.title('DFT')
        plt.show()

        # moving average
        ma = y[0]
        mv_avg = []
        w_ma = 0.5
        for i in range(len(y)):
            ma = y[0] if i==0 else w_ma*ma + (1-w_ma)*y[i]
            mv_avg.append(ma)
        self.logger.info('MA ' + str(mv_avg))
        plt.plot(t, mv_avg, marker='o', linestyle='-', color='b', label='Moving Average')
        plt.title('MA')
        plt.show()

        # Modeling absolute series

        # Modeling differences
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)

    TsDecomposeUnitTest(logger=lgr).test()
    exit(0)
