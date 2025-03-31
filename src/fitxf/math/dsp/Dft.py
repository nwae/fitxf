import logging
import numpy as np
import matplotlib.pyplot as plt
from fitxf.utils import Logging
from fitxf.math.utils.LoggingSingleton import LoggingSingleton


class Dft:

    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def getSineSignal(
            self,
            amps: np.ndarray,
            freqs: np.ndarray,
            samp_rate: float,
            show_plot: bool = False,
    ):
        assert len(amps) == len(freqs)

        dt = 1.0 / samp_rate
        t_start = 0
        t_end = 1
        t = np.arange(t_start, t_end, dt)

        x = np.zeros_like(t)
        for a, f in zip(amps, freqs):
            x += a * np.sin(2 * np.pi * f * t)

        if show_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(t, x, 'r')
            plt.xlabel('t')
            plt.ylabel('x')
            plt.grid(True)
            plt.show()

        return x

    def DFT(
            self,
            # time domain signal
            x: np.ndarray,
            lib: str = 'dot'
    ):
        if lib == 'numpy':
            return np.fft.fft(x)
        else:
            N = x.size
            # n is indexed at the columns
            n_vec = np.arange(N)
            # reshape so that k is indexed at the rows
            k_vec = n_vec.reshape((N, 1))
            # print(k_vec)
            # raise Exception('asdf')

            e = np.exp(-2j * np.pi * k_vec * n_vec / N)
            # print(e, e.shape)
            # raise Exception('asdf')
            return np.dot(e, x)

    def IDFT(
            self,
            y: np.ndarray,
    ):
        N = y.size
        # k is indexed at the columns
        k_vec = np.arange(N)
        # reshape so that n is indexed at the rows
        n_vec = k_vec.reshape((N, 1))
        # print(k_vec)
        # raise Exception('asdf')

        e = np.exp(2j * np.pi * k_vec * n_vec / N)
        # print(e, e.shape)
        # raise Exception('asdf')
        return (1 / N) * np.dot(e, y)

    def plot_fft(self, X_fft: np.ndarray, freq: np.ndarray):
        plt.figure(figsize=(8,6))
        plt.stem(freq, abs(X_fft), 'b', markerfmt=' ', basefmt='-b')
        plt.xlabel(r'$\omega$ '+'(Hz)')
        plt.ylabel('DFT Amplitude '+r'$|X(\omega)|$')
        plt.grid(True)
        plt.show()
        return


class DftUnitTest:

    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()

    def test(self):
        dft = Dft(logger=self.logger)

        x = np.random.random(100)

        y = dft.DFT(x=x)
        idft = dft.IDFT(y=y)
        err = idft - x
        abs_err = abs(err)
        assert np.sum(abs_err**2) < 0.0000000001
        return


if __name__ == '__main__':
    lgr = LoggingSingleton.get_singleton_logger()
    DftUnitTest(logger=lgr).test()

    # sampling rate 100Hz
    rate = 30

    dft = Dft()

    amps = np.array([1, 2, 10])
    freqs = np.array([1, 2, 10])
    x = dft.getSineSignal(
        samp_rate=rate,
        amps=amps,
        freqs=freqs,
    )
    lgr.info('x shape ' + str(x.shape) + ': ' + str(x))

    X_k = dft.DFT(x=x)
    lgr.info('DFT ' + str(abs(X_k)) + ', shape ' + str(X_k.shape))
    N = X_k.size
    n = np.arange(N)
    # total time span
    T = N / rate
    freq = n / T
    lgr.info('N=' + str(N) + ', total time span T=' + str(T))
    lgr.info('Discrete freq: ' + str(freq))

    dft.plot_fft(X_k, freq=freq)
    idft = dft.IDFT(y=X_k)
    lgr.info('Inverse DFT error ' + str(idft - x))
    exit(0)

