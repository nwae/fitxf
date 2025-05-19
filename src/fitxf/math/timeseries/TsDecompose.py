import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fitxf.math.dsp.Dft import Dft
# from statsmodels.tsa.seasonal import seasonal_decompose
from fitxf.utils import Logging


#
# https://en.wikipedia.org/wiki/Decomposition_of_time_series
# https://www.ibm.com/think/topics/arima-model#:~:text=ARIMA%20stands%20for%20Autoregressive%20Integrated,to%20forecasting%20time%20series%20data.
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
            prepend_value: float = None,
            method: str = 'np',
    ):
        assert (p > 0) and (p < 1)
        q = 1 - p
        n = int(np.ceil( np.log(min_weight) / np.log(q) ))

        weights = p * np.array([q**k for k in range(n)])
        self.logger.info('Exponential MA weights for p = ' + str(p) + ', n = ' + str(n) + ': ' + str(weights))
        return self.calculate_ma(series=series, weights=weights, prepend_value=prepend_value, method=method)

    #
    # Moving average
    #
    def calculate_ma(
            self,
            series: np.ndarray,
            weights: np.ndarray,
            prepend_value: float = None,
            method: str = 'np',
    ) -> np.ndarray:
        assert series.ndim == 1
        assert len(weights) >= 2
        prepend_value = series[0] if prepend_value is None else prepend_value
        l_pp = len(weights) - 1
        # prepend with value of first index, so that MA at first index is the value itself
        preprend = np.array([prepend_value] * l_pp)
        series_x = np.append(preprend, series)
        if method == 'np':
            # numpy convolve will extend the length of the original series, so we clip it
            ma_tmp = np.convolve(a=series_x, v=weights, mode='full')[:len(series_x)]
            return ma_tmp[l_pp:]
        else:
            # manually calculate, mainly for unit tests
            inv_w = np.flip(weights, axis=0)
            self.logger.info('Flipped weights ' + str(inv_w) + ' prepended series ' + str(series_x))
            ma_tmp = np.array([
                np.sum(inv_w * series_x[i:(i + l_pp + 1)])
                for i in range(len(series))
            ])
            return ma_tmp

    #
    # cor(x, y) = E[(X-mu_x)(Y-mu_y)] / (sigma_x * sigma_y)
    #
    def calculate_correlation(
            self,
            x: np.ndarray,
            y: np.ndarray,
            # can be moving average or simple average, etc.
            x_mu: np.ndarray = None,
            y_mu: np.ndarray = None,
            var_x: float = 1.0,
            var_y: float = 1.0,
            # divide by correct lengths
            normalize_divide_lengths: bool = False,
            method: str = 'np',
    ):
        x_mu = 0.0 if x_mu is None else x_mu
        y_mu = 0.0 if y_mu is None else y_mu

        x_norm = x - x_mu
        y_norm = y - y_mu

        l_y = len(y_norm)
        l_extend = l_y - 1
        if normalize_divide_lengths:
            l_actual = np.append(
                np.array([len(y)] * l_extend),
                np.minimum(np.arange(len(x))[::-1] + 1, np.arange(len(y))[::-1] + 1),
                axis = 0,
            )
            # self.logger.info('Actual lengths ' + str(l_actual))
        else:
            l_actual = 1

        if method == 'np':
            # numpy correlation does not divide by length
            cor = np.correlate(x_norm, y_norm, mode="full") / l_actual
        else:
            # manually calculate, mainly for unit tests
            x_extended = np.array([0.0]*l_extend + x_norm.tolist() + [0.0]*l_extend)
            # self.logger.info('x extended ' + str(x_extended) + ', y ' + str(y_norm))
            cor = np.array([
                np.sum(y_norm * x_extended[i:(i + l_extend + 1)])
                for i in range(len(x_norm) + l_extend)
            ]) / l_actual
        return {int(idx): float(v) for idx, v in list(zip(np.arange(len(cor))-l_extend, cor))}
        # return cor[l_extend:] / (var_x * var_y)

    def calculate_auto_correlation(
            self,
            x: np.ndarray,
            # can be moving average or simple average, etc.
            # however, it is sometimes more accurate to just use simple average to calculate
            # auto-correlation.
            x_mu: np.ndarray,
            normalize_divide_lengths: bool = False,
            method: str = 'np',
    ):
        return self.calculate_correlation(
            x = x,
            y = x,
            x_mu = x_mu,
            y_mu = x_mu,
            var_x = float(np.var(x)),
            var_y = float(np.var(x)),
            normalize_divide_lengths = normalize_divide_lengths,
            method = method,
        )

    def correct_seasonality(
            self,
            x: np.ndarray,
            # can be moving average or simple average, etc.
            # however, it is sometimes more accurate to just use simple average to calculate
            # auto-correlation.
            x_mu: np.ndarray,
            method: str = 'np',
    ):
        # we only check auto-correlation for shifts of 1 to len_max
        len_max = int(len(x) / 2)
        self.logger.info('Check for lengths up to ' + str(len_max) + ' from total x length ' + str(len(x)))
        ac_np_dict = self.calculate_auto_correlation(x=x, x_mu=x_mu, normalize_divide_lengths=False)
        ac_np = np.array([v for i, v in ac_np_dict.items() if (i >= 1) and (i <= len_max)])
        idxs_sorted = np.argsort(a=ac_np, axis=-1)[::-1]
        ac_np_sorted = ac_np[idxs_sorted]
        top_index_ac = ac_np_sorted[0]
        self.logger.info(
            'Top index ' + str(top_index_ac) + ', auto correlation ' + str(ac_np_sorted)
            + ', sorted indexes ' + str(idxs_sorted)
        )
        # correct the seasonality

        raise Exception('asdf')
        return


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
        exp_ma_pp_none = np.array([5., 7.5, 8.7, 7.7, 6.7, 5.7, 4.7, 3.7, 2.7, 1.7])
        exp_ma_pp      = np.array([10., 9.5, 8.7, 7.7, 6.7, 5.7, 4.7, 3.7, 2.7, 1.7])
        for i, (ser, w, prepend_val, exp_ma) in enumerate([
            (series, weights, 0.0, exp_ma_pp_none,),
            (series, weights, series[0], exp_ma_pp,),
        ]):
            ma_mn = ts_dec.calculate_ma(series=ser, weights=w, prepend_value=prepend_val, method='manual')
            ma_np = ts_dec.calculate_ma(series=ser, weights=w, prepend_value=prepend_val, method='np')
            self.logger.info('# ' + str(i) + ' MA manual: ' + str(ma_mn) + '\n, via numpy: ' + str(ma_np))
            assert np.sum((ma_mn - ma_np)**2) < 0.0000000001, \
                '# ' + str(i) + ' MA manual ' + str(ma_mn) + ' not ' + str(ma_np)
            assert np.sum((ma_mn - exp_ma)**2) < 0.0000000001, \
                '# ' + str(i) + ' MA manual ' + str(ma_mn) + ' not ' + str(exp_ma)
            assert np.sum((ma_np - exp_ma)**2) < 0.0000000001, \
                '# ' + str(i) + ' MA numpy ' + str(ma_np) + ' not ' + str(exp_ma)

        #
        # Test exponential MA
        #
        exp_ma_exp_pp_none = np.array([4., 6., 6.8, 6.88, 6.528, 5.9168, 5.15008, 4.290048, 3.3740288, 2.42441728])
        exp_ma_exp_pp = np.array([9.99999386, 9.59999386, 8.95999386, 8.17599386, 7.30559386,
                                  6.38335386, 5.43000986, 4.45800346, 3.47479962, 2.48487732])
        for i, (ser, prepend_val, exp_ma) in enumerate([
            (series, 0.0, exp_ma_exp_pp_none,),
            (series, series[0], exp_ma_exp_pp,),
        ]):
            ma_exp = ts_dec.calculate_ma_exponential(series=ser, p=0.4, prepend_value=prepend_val)
            self.logger.info('MA exp: ' + str(ma_exp))
            assert np.sum((ma_exp - exp_ma)**2) < 0.0000000001, \
                '#' + str(i) + ' MA exponential ' + str(ma_exp) + ' not ' + str(exp_ma)

        #
        # Test Correlation
        #
        x = np.array([1, 2, 3])
        y = np.array([0, 1, 0.5])
        len_actual = np.array([3, 2, 1])
        exp_cor_no_div = np.array([3.5, 3.,  0. ])
        exp_cor_div = exp_cor_no_div / len_actual
        for norm_div_len, exp_cor in [(False, exp_cor_no_div), (True, exp_cor_div)]:
            cor_np = ts_dec.calculate_correlation(x=x, y=y, normalize_divide_lengths=norm_div_len, method='np')
            cor_mn = ts_dec.calculate_correlation(x=x, y=y, normalize_divide_lengths=norm_div_len, method='manual')
            self.logger.info(
                'Normalize divide lengths ' + str(norm_div_len) + ', correlation ' + str(cor_np)
                + ', manual ' + str(cor_mn)
            )
            err_np = np.sum(([v for i, v in cor_np.items() if i >= 0] - exp_cor)**2)
            err_mn = np.sum(([v for i, v in cor_mn.items() if i >= 0] - exp_cor)**2)
            assert err_np < 0.0000000001, 'Cor numpy ' + str(cor_np) + ' not ' + str(exp_cor)
            assert err_mn < 0.0000000001, 'Cor manual ' + str(cor_mn) + ' not ' + str(exp_cor)

        #
        # Test Auto-Correlation
        #
        # seasonality at 2nd index
        x = np.array([1, 2, 10, 2, 3, 11, 1, 4, 9, 2, 2, 11])
        len_actual = np.array([12, 11, 10,  9,  8 , 7,  6,  5,  4,  3,  2,  1,])
        # Seasonality at index shift +3
        exp_seasonality = 3
        x_mu = float(np.mean(x))
        self.logger.info('MA for x ' + str(x_mu))
        ac_np_dict = ts_dec.calculate_auto_correlation(x=x, x_mu=x_mu, normalize_divide_lengths=False)
        ac_np = np.array([v for i, v in ac_np_dict.items() if i >= 0])
        max_ac_idx = np.argsort(ac_np, axis=-1)
        self.logger.info('Auto-correlation ' + str(ac_np) + ', max AC index ' + str(max_ac_idx))
        seasonality_n = max_ac_idx[-2]
        # The biggest auto-correlation is when there is no shift
        assert max_ac_idx[-1] == 0
        assert seasonality_n == exp_seasonality, \
            'Seasonality at period ' + str(seasonality_n) + ' not ' + str(exp_seasonality)

        #
        # Test seasonality correction
        #
        ts_dec.correct_seasonality(x=x, x_mu=x_mu)

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
