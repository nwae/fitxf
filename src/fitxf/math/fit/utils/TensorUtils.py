import numpy as np
import torch
import warnings
import logging
from fitxf.utils import Logging


class TensorUtils:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        # ignore divide by 0 warnings
        warnings.filterwarnings('ignore')
        return

    def convert_to_numpy(self, x):
        assert type(x) in [np.ndarray, torch.Tensor], 'Wrong type "' + str(type(x)) + '": ' + str(x)
        return x.cpu().detach().numpy() if type(x) is torch.Tensor else x

    """
    Normalize Tensor of any dimension
    """
    def normalize(
            self,
            x,
            return_tensors = 'np',
    ):
        x_tmp = self.convert_to_numpy(x=x)

        # normalize the last row dimension
        norm = np.linalg.norm(x_tmp, axis=-1, keepdims=True)
        x_norm = x_tmp / norm

        # handle 0's
        x_norm[np.isnan(x_norm)] = 0.
        return torch.from_numpy(x_norm) if return_tensors=='pt' else x_norm

    def similarity_cosine(self, x, ref, return_tensors='np'):
        return self.dot_sim(x=x, ref=ref, return_tensors=return_tensors)

    # arccos similarity
    def dot_sim(
            self,
            x,
            ref,
            return_tensors = 'np',
    ):
        x_np = self.convert_to_numpy(x=x)
        ref_np = self.convert_to_numpy(x=ref)
        assert x_np.ndim == 2, 'Dim must be 2 but got ' + str(x_np.ndim) + ': ' + str(x_np)
        assert ref_np.ndim == 2, 'Dim must be 2 but got ' + str(ref_np.ndim) + ': ' + str(ref_np)
        x_norm = self.normalize(
            x = x_np,
            return_tensors = 'np',
        )
        ref_norm = self.normalize(
            x = ref_np,
            return_tensors = 'np',
        )
        self.logger.debug('t shape ' + str(ref_norm.shape) + ', txt_lm shape ' + str(x_norm.shape))
        # can be negative which means in the opposite direction from reference
        m_dot = np.matmul(x_norm, np.transpose(ref_norm))
        # Avoid nan for arccos for values 1+smallvalue or -1-smallvalue
        # values returned mapped from cosine [-1,1] range, which corresponds ranges from [0, pi],
        # never negative unlike arcsin()
        # m_dot = np.arccos(np.minimum(np.maximum(m_dot, -1.), 1.))
        # sort descending, flip the argsort which by default is ascending
        result_ordered = np.flip( np.argsort(m_dot, axis=-1), axis=-1 )
        m_dot_ordered = np.array([
            [m_dot[i,result_ordered[i,j]] for j in range(m_dot.shape[1])] for i in range(m_dot.shape[0])
        ])

        return (torch.from_numpy(result_ordered.copy()), torch.from_numpy(m_dot_ordered.copy())) \
            if return_tensors == 'pt' else (result_ordered, m_dot_ordered)

    def similarity_distance(
            self,
            x,
            ref,
            return_tensors = 'np',
    ):
        x_np = self.convert_to_numpy(x=x)
        ref_np = self.convert_to_numpy(x=ref)
        assert x_np.ndim == 2, 'Dim must be 2 but got ' + str(x_np.ndim) + ': ' + str(x_np)
        assert ref_np.ndim == 2, 'Dim must be 2 but got ' + str(ref_np.ndim) + ': ' + str(ref_np)

        # Make compatible for taking difference
        # Convert for example row vectors
        #    [
        #      [1,2,3],
        #      [4,5,6],
        #    ]
        # into a repeated sequence (same count as how many rows in reference tensor)
        #    [
        #      [ [1,2,3], [1,2,3], [1,2,3] ],
        #      [ [4,5,6], [4,5,6], [4,5,6] ],
        #    ]
        self.logger.debug('x:\n' + str(x_np))
        self.logger.debug('ref:\n' + str(ref_np))
        x_np_rp = np.repeat(x_np, len(ref_np), axis=0)
        x_np_rp = np.reshape(x_np_rp, (len(x_np), len(ref_np), x_np.shape[-1]))
        self.logger.debug('x repeated:\n' + str(x_np_rp))
        m_difsq = (x_np_rp - ref_np) ** 2
        self.logger.debug('difference squared:\n' + str(m_difsq))
        m_dist = np.sum(m_difsq, axis=-1) ** 0.5
        self.logger.debug('distances:\n' + str(m_dist))
        result_ordered = np.argsort(m_dist, axis=-1)
        self.logger.debug('index ordered:\n' + str(result_ordered))
        m_dist_ordered = np.array([
            [m_dist[i,result_ordered[i,j]] for j in range(m_dist.shape[1])] for i in range(m_dist.shape[0])
        ])
        self.logger.debug('distance ordered:\n' + str(m_dist_ordered))

        return (torch.from_numpy(result_ordered.copy()), torch.from_numpy(m_dist_ordered.copy())) \
            if return_tensors == 'pt' else (result_ordered, m_dist_ordered)

    def hamming_distance_int(
            self,
            n1: int,
            n2: int,
            method: str | None = None,
    ) -> int:
        xor_result = n1 ^ n2
        self.logger.debug('xor result between ' + str(bin(n1)) + ' & ' + str(bin(n2)) + ' = ' + str(bin(xor_result)))
        # Count set bits (ones) in the XOR result
        count = 0
        while xor_result > 0:
            xor_result &= (xor_result - 1)  # Brian Kernighan's algorithm
            count += 1
        return count

    def convert_binary_encoding_to_int_vect(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 1
        # this gives us [128, 64, 32, 16,  8,  4,  2,  1]
        # ref = np.array([2], dtype=np.uint8) ** np.arange(8)[::-1]
        ref = np.array([128, 64, 32, 16,  8,  4,  2,  1])
        v = np.vstack([1*(v&ref > 0) for v in x]).reshape(len(x)*len(ref))
        return v.astype(np.int8)

    def convert_binary_encoding_to_int(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim in [1, 2]
        if x.ndim == 1:
            return self.convert_binary_encoding_to_int_vect(x=x)
        else:
            return np.array([self.convert_binary_encoding_to_int_vect(x=v) for v in x], dtype=np.int8)


class TensorUtilsUnitTest:

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def _helper_test_numpy_arrays(self, x1, x2):
        assert x1.shape == x2.shape
        dif = np.sum((x1 - x2)**2)
        assert dif < 0.0000000001

    def test_norm(self):
        m = TensorUtils()

        v_np_1d = np.array([1, 2, 3, 4])
        expected = np.array([0.18257419, 0.36514837, 0.54772256, 0.73029674])
        nm = m.normalize(x=v_np_1d, return_tensors='np')
        self._helper_test_numpy_arrays(x1=nm, x2=expected)

        v_np = np.array([[1, 2, 3], [5, 4, 6], [-1, -2, -3], [0, 0, 0]])
        v_pt = torch.FloatTensor([[1, 2, 3], [5, 4, 6], [-1, -2, -3], [0, 0, 0]])
        expected = np.array([
            [ 0.26726124,  0.53452248,  0.80178373],
            [ 0.56980288,  0.45584231,  0.68376346],
            [-0.26726124, -0.53452248, -0.80178373],
            [ 0.,          0.,          0.,        ],
        ])
        nm = m.normalize(x=v_np, return_tensors='np')
        self._helper_test_numpy_arrays(x1=nm, x2=expected)
        nm = m.normalize(x=v_pt, return_tensors='pt')
        self._helper_test_numpy_arrays(x1=nm.cpu().detach().numpy(), x2=expected)

        v_np_3d = np.array([
            [[1, 2, 3], [5, 4, 6], [-1, -2, -3], [0, 0, 0]],
            [[1, 1, 1], [2, 2, 2], [-1, -1, -1], [0, 0, 0]],
        ])
        expected = np.array([
            [
                [ 0.26726124,  0.53452248,  0.80178373],
                [ 0.56980288,  0.45584231,  0.68376346],
                [-0.26726124, -0.53452248, -0.80178373],
                [ 0.,          0.,          0.,        ],
            ],
            [
                [ 0.57735027,  0.57735027,  0.57735027],
                [ 0.57735027,  0.57735027,  0.57735027],
                [-0.57735027, -0.57735027, -0.57735027],
                [ 0.,          0.,          0.,        ],
            ],
        ])
        nm = m.normalize(v_np_3d)
        self._helper_test_numpy_arrays(x1=nm, x2=expected)
        self.logger.info('ALL TESTS PASSED OK (norm)')
        return

    def test_similarity_cosine_and_similarity_distance(self):
        mu = TensorUtils()
        x = np.array([
            [1, 2, 3],
            [6, 5, 4],
        ])
        ref = np.array([
            [1.2, 1.9, 3.3],
            [6.7, 4.8, 4.1],
            [-6.7, 4.8, -4.1],
        ])
        res, m = mu.dot_sim(
            x = x,
            ref = ref,
            return_tensors = 'np',
        )
        expected_res = np.array([[0, 1, 2], [1, 0, 2]])
        expected_m = np.array([[0.99742005,  0.83034349, -0.2729101], [0.99780448,  0.85345704, -0.40357849]])
        self._helper_test_numpy_arrays(x1=res, x2=expected_res)
        self._helper_test_numpy_arrays(x1=m, x2=expected_m)

        res, m = mu.similarity_distance(
            x = x,
            ref = ref,
            return_tensors = 'np',
        )
        expected_res = np.array([[0, 1, 2], [1, 0, 2]])
        expected_m = np.array([[0.37416574,  6.44515322, 10.8415866], [0.73484692,  5.75673519, 15.06452787]])
        self._helper_test_numpy_arrays(x1=res, x2=expected_res)
        self._helper_test_numpy_arrays(x1=m, x2=expected_m)
        self.logger.info('ALL TESTS PASSED OK (similarity cosine & distance)')
        return

    def test_hamming_distance(self):
        tu = TensorUtils(logger=self.logger)
        x_raw = np.array([0b10101010, 0b00001111])
        x_arr = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        y_raw = np.array([0b00111010, 0b01001100])
        y_arr = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0])

        x_01 = tu.convert_binary_encoding_to_int(x=x_raw)
        assert x_01.dtype == np.int8
        assert x_01.nbytes == 16
        assert np.sum((x_01 - x_arr) ** 2) < 0.0000000001, 'Got x ' + str(x_01) + ' but expect ' + str(x_arr)
        y_01 = tu.convert_binary_encoding_to_int(x=y_raw)
        assert y_01.dtype == np.int8
        assert y_01.nbytes == 16
        assert np.sum((y_01 - y_arr) ** 2) < 0.0000000001, 'Got y ' + str(y_01) + ' but expect ' + str(y_arr)

        x2d = np.array([[0b10101010, 0b00001111], [0b00111010, 0b01001100]])
        x2d_arr = np.array([
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]],
        )
        x2d_01 = tu.convert_binary_encoding_to_int(x=x2d)
        assert np.sum((x2d_01 - x2d_arr) ** 2) < 0.0000000001, \
            '2D test got x2d ' + str(x2d_01) + ' but expect ' + str(x2d_arr)

        for dtyp in [np.uint8]:
            x = x_raw.astype(dtyp)
            y = y_raw.astype(dtyp)
            exp_dists = [2, 3]
            for i in range(len(x)):
                d = tu.hamming_distance_int(n1=x[i], n2=y[i])
                self.logger.info(
                    'Test dtype ' + str(dtyp) + ' #' + str(i) + ' distance of ' + str(bin(x[i]))
                    + ' & ' + str(bin(y[i])) + ' = ' + str(d)
                )
                assert d == exp_dists[i], \
                    'Test dtype ' + str(dtyp) + ' #' + str(i) + ' between ' + str(bin(x[i])) + ' & ' + str(bin(y[i]))\
                    + ' = ' + str(d) + ' but expected ' + str(exp_dists[i])
        self.logger.info('Hamming TESTS PASSED')
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    ut = TensorUtilsUnitTest(logger=lgr)
    ut.test_norm()
    ut.test_similarity_cosine_and_similarity_distance()
    ut.test_hamming_distance()
    exit(0)
