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
    # q = 1 - p
    # ma[T] = px[T] + qx[T-1]
    #       = px[T] + pq x[T-1] + qq x[T-2]
    #       = px[T] + pq x[T-1] + pqq x[T-2] + ... + pq^n x[T-n] + q^(n+1) x[T-n-1]
    #
    def calculate_ma_exponential(
            self,
            series: np.ndarray,
            p: float,
            min_weight: float = 0.000001,
            method: str = 'np',
    ):
        assert (p > 0) and (p < 1)
        q = 1 - p
        n = int(np.ceil( np.log(min_weight) / np.log(q) ))

        weights = p * np.array([q**k for k in range(n)])
        self.logger.info('Exponential MA weights for p = ' + str(p) + ', n = ' + str(n) + ': ' + str(weights))
        return self.calculate_ma(series=series, weights=weights, method=method)

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
        assert len(weights) >= 2
        if method == 'np':
            # numpy convolve will extend the length of the original series, so we clip it
            return np.convolve(a=series, v=weights, mode='full')[:len(series)]
        else:
            # manually calculate, mainly for unit tests
            inv_w = np.flip(weights, axis=0)
            self.logger.info('Flipped weights ' + str(inv_w))
            l_add = len(weights) - 1
            series_extended = np.array([0.0]*l_add + series.tolist())
            self.logger.info('Series extended ' + str(series_extended))
            return np.array([np.sum(inv_w * series_extended[i:(i + l_add + 1)]) for i in range(len(series))])

    def calculate_correlation(
            self,
            x: np.ndarray,
            y: np.ndarray,
            # can be moving average or simple average, etc.
            x_mu: np.ndarray = None,
            y_mu: np.ndarray = None,
            var_x: float = 1.0,
            var_y: float = 1.0,
            method: str = 'np',
    ):
        x_mu = 0.0 if x_mu is None else x_mu
        y_mu = 0.0 if y_mu is None else y_mu

        x_norm = x - x_mu
        y_norm = y - y_mu

        l_extend = len(y_norm) - 1

        if method == 'np':
            cor = np.correlate(x_norm, y_norm, mode="full")
            return cor / (var_x * var_y)
        else:
            # manually calculate, mainly for unit tests
            x_extended = np.array([0.0]*l_extend + x_norm.tolist() + [0.0]*l_extend)
            self.logger.info('x extended ' + str(x_extended) + ', y ' + str(y_norm))
            cor = np.array([
                np.sum(y_norm* x_extended[i:(i + l_extend + 1)])
                for i in range(len(x_norm) + l_extend)
            ])
            return cor / (var_x * var_y)

    def calculate_auto_correlation(
            self,
            x: np.ndarray,
            # can be moving average or simple average, etc.
            x_mu: np.ndarray,
            method: str = 'np',
    ):
        auto_cor = self.calculate_correlation(
            x = x,
            y = x,
            x_mu = x_mu,
            y_mu = x_mu,
            var_x = float(np.var(x)),
            var_y = float(np.var(x)),
            method = method,
        )
        l_extend = len(x) - 1
        l_end = l_extend + len(x)
        auto_cor_0 = auto_cor[l_extend:l_end]
        return auto_cor_0


class TsDecomposeUnitTest:
    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        ts_dec = TsDecompose(logger=self.logger)

        #
        # Test MA
        #
        N = 10
        series = np.flip(np.arange(N) + 1, axis=0).astype(np.float32)
        self.logger.info('Series: ' + str(series))
        weights = np.array([0.5, 0.3, 0.2])
        exp_ma = np.array([5., 7.5, 8.7, 7.7, 6.7, 5.7, 4.7, 3.7, 2.7, 1.7])
        ma_mn = ts_dec.calculate_ma(series=series, weights=weights, method='manual')
        ma_np = ts_dec.calculate_ma(series=series, weights=weights, method='np')
        self.logger.info('MA manual: ' + str(ma_mn) + '\n, via numpy: ' + str(ma_np))
        assert np.sum((ma_mn**2) - (ma_np**2)) < 0.0000000001, 'MA manual ' + str(ma_mn) + ' not ' + str(ma_np)
        assert np.sum((ma_mn**2) - (exp_ma**2)) < 0.0000000001, 'MA manual ' + str(ma_mn) + ' not ' + str(exp_ma)
        assert np.sum((ma_np**2) - (exp_ma**2)) < 0.0000000001, 'MA numpy ' + str(ma_np) + ' not ' + str(exp_ma)

        #
        # Test exponential MA
        #
        ma_exp = ts_dec.calculate_ma_exponential(series=series, p=0.4)
        exp_ma_exp = np.array([4., 6., 6.8, 6.88, 6.528, 5.9168, 5.15008, 4.290048, 3.3740288, 2.42441728])
        self.logger.info('MA exp: ' + str(ma_exp))
        assert np.sum((ma_exp**2) - (exp_ma_exp**2)) < 0.0000000001, \
            'MA exponential ' + str(ma_exp) + ' not ' + str(exp_ma_exp)

        #
        # Test Correlation
        #
        x = np.array([1, 2, 3])
        y = np.array([0, 1, 0.5])
        exp_cor = np.array([0.5, 2.,  3.5, 3.,  0. ])
        cor_np = ts_dec.calculate_correlation(x=x, y=y, method='np')
        cor_mn = ts_dec.calculate_correlation(x=x, y=y, method='manual')
        self.logger.info('Correlation ' + str(cor_np) + ', manual ' + str(cor_mn))
        assert np.sum((cor_np**2) - (exp_cor**2)) < 0.0000000001, 'Cor numpy ' + str(cor_np) + ' not ' + str(exp_cor)
        assert np.sum((cor_mn**2) - (exp_cor**2)) < 0.0000000001, 'Cor manual ' + str(cor_mn) + ' not ' + str(exp_cor)

        #
        # Test Auto-Correlation
        #
        x = np.array([1, 2, 3, 10, 3, 2, 12, 2, 2])
        # Seasonality at index shift +3
        exp_seasonality = 3
        x_mu = float(np.mean(x))
        self.logger.info('MA for x ' + str(x_mu))
        ac_np = ts_dec.calculate_auto_correlation(x=x, x_mu=x_mu)
        max_ac_idx = np.argsort(ac_np, axis=-1)
        seasonality_n = max_ac_idx[-2]
        # The biggest auto-correlation is when there is no shift
        assert max_ac_idx[-1] == 0
        self.logger.info('Auto-correlation ' + str(ac_np) + ', max AC index ' + str(max_ac_idx))
        assert seasonality_n == exp_seasonality, \
            'Seasonality at period ' + str(seasonality_n) + ' not ' + str(exp_seasonality)

        #
        # Generate random time series, with cycle of sine
        #
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

        self.logger.info('Tests Passed')
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)

    TsDecomposeUnitTest(logger=lgr).test()
    exit(0)
