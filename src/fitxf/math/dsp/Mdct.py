import logging
import numpy as np
import matplotlib.pyplot as plt
from fitxf.utils import Logging
from fitxf.math.utils.LoggingSingleton import LoggingSingleton


class Mdct:

    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def getSineSignal(
            self,
            amps,
            freqs,
            t: np.ndarray,
            show_plot: bool = False,
    ):
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

    def MDCT(
            self,
            # time domain signal
            x: np.ndarray,
            lib: str = 'dot'
    ):
        assert len(x) % 2 == 0
        N = int(len(x) / 2)
        # n is indexed at the columns
        n = np.arange(2 * N)
        # print(n)
        # reshape so that k is indexed at the rows
        k = np.reshape(np.arange(N), shape=(N, 1))
        # print(k)
        A = 0.5 + N/2
        B = 0.5
        # now we have a k x n matrix
        m = np.cos((np.pi / N) * (n + A) * (k + B))
        # x_square = np.array([x.tolist() for i in range(N)])
        # [print(line) for line in np.round(m,3).tolist()]
        y = np.dot(m, x)
        return y

    def IMDCT(
            self,
            y: np.ndarray,
    ):
        N = len(y)
        # k is indexed at the columns
        k = np.arange(N)
        # print(n)
        # reshape so that n is indexed at the rows
        n = np.reshape(np.arange(2 * N), shape=(2 * N, 1))
        # print(k)
        A = 0.5 + N/2
        B = 0.5
        # now we have a k x n matrix
        m = (1 / N) * np.cos((np.pi / N) * (n + A) * (k + B))
        # T_square = np.array([T.tolist() for i in range(2*N)])
        # [print(line) for line in np.round(m,3).tolist()]
        imdct = np.dot(m, y)
        return imdct

    def test_sum(
            self,
            x: np.ndarray,
            k1: int,
            k2: int,
            f,
    ):
        assert len(x) % 2 == 0
        N = int(len(x) / 2)
        # n is indexed at the columns
        n = np.arange(2 * N)

        A = 0.5 + N/2
        B = 0.5
        m = f((np.pi / N) * (n + A) * (k1 + B)) * f((np.pi / N) * (n + A) * (k2 + B))
        self.logger.debug(m)
        v = np.dot(m, x) / N
        return v


class MdctUnitTest:

    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()

    def test(self):
        tfm = Mdct(logger=self.logger)

        x = np.random.random(100)
        self.logger.debug(x)
        y = tfm.MDCT(x=x)
        imdct = tfm.IMDCT(y=y)
        self.logger.debug(imdct)
        err = imdct - x
        abs_err = abs(err)
        assert np.sum(abs_err**2) < 0.0000000001
        return


if __name__ == '__main__':
    lgr = LoggingSingleton.get_singleton_logger(log_level=logging.DEBUG)
    obj = Mdct(logger=lgr)
    t_val = obj.test_sum(x=np.ones(10), k1=2, k2=65, f=np.sin)
    lgr.info(t_val)
    exit(0)

    MdctUnitTest(logger=lgr).test()
    exit(0)

