import logging
import numpy as np
from fitxf.math.utils.Logging import Logging


class MathUtils:
  
    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def match_template_1d(
            self,
            x: np.ndarray,
            seq: np.ndarray,
    ) -> list:
        x = np.array(x) if type(x) in (list, tuple) else x
        seq = np.array(seq) if type(seq) in (list, tuple) else seq
        assert x.ndim == seq.ndim == 1
        assert len(seq) <= len(x)

        l_x, l_seq = len(x), len(seq)

        # Template for sequence
        r_seq = np.arange(l_seq)

        # Create the template matching indices in 2D. e.g.
        #   [
        #     [0],
        #     [1],
        #     [2],
        #     ...
        #     [n],
        #     ...
        #   ]
        template_matching_indices = np.arange(l_x - l_seq + 1)[:, None]
        # self.logger.debug('Template matching indices 2D structure: ' + str(template_matching_indices))
        # Create the template matching indices in 2D. e.g. for seq length 3
        #   [
        #     [0,1,2],
        #     [1,2,3],
        #     [2,3,4],
        #     ...
        #     [n,n+1,n+2],
        #     ...
        #   ]
        template_matching_indices = template_matching_indices + r_seq
        self.logger.debug('Template matching indices final: ' + str(template_matching_indices))
        # Find matches
        template_matches = x[template_matching_indices] == seq
        self.logger.debug('Template matches for seq ' + str(seq) + ': ' + str(template_matches))

        #
        # nan means "*" match like in string regex
        #
        nan_positions = np.isnan(seq)
        self.logger.debug('nan positions: ' + str(nan_positions))
        template_matches = 1 * (template_matches | nan_positions)
        self.logger.debug('Template matches with nan for seq ' + str(seq) + ': ' + str(template_matches))

        # Match is when all are 1's
        match_start_indexes = 1 * (np.sum(template_matches, axis=-1) == len(seq))
        self.logger.debug('Match start indexes: ' + str(match_start_indexes))

        # Get the range of those indices as final output
        if match_start_indexes.any() > 0:
            res =  np.argwhere(match_start_indexes == 1).flatten().tolist()
            # return {
            #     'match_indexes': np.where(match_indexes == 1)[0],
            #     'match_sequence': np.where(np.convolve(match_indexes, np.ones((Nseq), dtype=int)) > 0)[0]
            # }
        else:
            res = []
            # return {
            #     'match_indexes': [],
            #     'match_sequence': [],  # No match found
            # }
        self.logger.info('Match indexes type "' + str(type(res)) + '": ' + str(res))
        return res

    def match_template(
            self,
            x: np.ndarray,
            seq: np.ndarray,
    ) -> list:
        x = np.array(x) if type(x) in (list, tuple) else x
        seq = np.array(seq) if type(seq) in (list, tuple) else seq
        assert x.ndim == seq.ndim, 'Dimensions do not match, x dim ' + str(x.ndim) + ', seq dim ' + str(seq.ndim)

        n_dim = x.ndim

        # for dim in range(x.ndim):
        #     assert len(seq[dim]) <= len(x[dim]), \
        #         'At dim #' + str(dim) + ', sequence length ' + str(len(seq[dim])) \
        #         + ' cannot be longer than x length ' + str(len(x[dim]))

        template_shape = [x.shape[i] - seq.shape[i] + 1 for i in range(n_dim)]
        self.logger.debug(
            'Template shape ' + str(template_shape) + ', sequence shape ' + str(seq.shape)
        )
        # l_x, l_seq = len(x), len(seq)

        # Template for sequence
        r_seq = np.array( np.arange(seq.shape[0]).tolist() * int(np.prod(seq.shape) / seq.shape[0]) )
        r_seq.resize(seq.shape)
        self.logger.debug('Sequence for matching ' + str(r_seq))

        # Create the template matching indices in 2D. e.g.
        #   [
        #     [0],
        #     [1],
        #     [2],
        #     ...
        #     [n],
        #     ...
        #   ]
        template_matching_indices = np.arange(template_shape[-1])
        new_size = [len(template_matching_indices)]
        for i in range(n_dim):
            new_size = new_size + [1]
            template_matching_indices.resize(new_size)
            self.logger.debug('New size ' + str(new_size))
        self.logger.debug('Template matching indices ' + str(n_dim) + 'D structure: ' + str(template_matching_indices))
        # Create the template matching indices in 2D. e.g. for seq length 3
        #   [
        #     [0,1,2],
        #     [1,2,3],
        #     [2,3,4],
        #     ...
        #     [n,n+1,n+2],
        #     ...
        #   ]
        template_matching_indices = template_matching_indices + r_seq
        self.logger.debug('Template matching indices final: ' + str(template_matching_indices))
        # Find matches
        self.logger.debug('x to index ' + str(x))
        self.logger.debug('   with template ' + str(x[template_matching_indices]))
        template_matches = 1 * (x[template_matching_indices] == seq)
        self.logger.debug('Template matches: ' + str(template_matches))
        match_start_indexes = 1 * (np.sum(template_matches, axis=-1) == len(seq))
        self.logger.debug('Match start indexes: ' + str(match_start_indexes))

        # Get the range of those indices as final output
        if match_start_indexes.any() > 0:
            res =  np.argwhere(match_start_indexes == 1).flatten().tolist()
            # return {
            #     'match_indexes': np.where(match_indexes == 1)[0],
            #     'match_sequence': np.where(np.convolve(match_indexes, np.ones((Nseq), dtype=int)) > 0)[0]
            # }
        else:
            res = []
            # return {
            #     'match_indexes': [],
            #     'match_sequence': [],  # No match found
            # }
        self.logger.info('Match indexes type "' + str(type(res)) + '": ' + str(res))
        return res

    def sample_random_no_repeat(
            self,
            list,
            n,
    ):
        assert n <= len(list)
        rng = np.random.default_rng()
        numbers = rng.choice(len(list), size=n, replace=False)
        sampled = []
        for i in numbers:
            sampled.append(list[i])
        return sampled


class MathUtilsUnitTest:
    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        self.test_1d()
        self.logger.info('ALL TESTS PASSED')
        return

    def test_1d(self):
        mu = MathUtils(logger=self.logger)

        # Test 1D
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        x = np.arange(20) % 10
        for seq, exp_matches in [
            (np.array([1, 2, 3, 4]), np.array([1, 11])),
            (np.array([1, np.nan, np.nan, 4]), np.array([1, 11])),
            (np.array([9, 10, 11]), np.array([9])),
            # (np.array([1, 3, 5]), []),
        ]:
            match_idxs = mu.match_template_1d(x=x, seq=seq)
            assert np.sum((np.array(match_idxs) - exp_matches)**2) < 0.0000000001, \
                'Match indexes ' + str(match_idxs) + ' not ' + str(exp_matches)
        return

    def test_2d(self):
        mu = MathUtils(logger=self.logger)

        # Test 2D
        # [[0 1 2 3 4]
        #  [5 6 7 8 9]
        #  [0 1 2 3 4]
        #  [5 6 7 8 9]]
        x = np.arange(20) % 10
        x.resize((4, 5))
        self.logger.info('2D test data:\n' + str(x))
        for seq, exp_matches in [
            (np.array([[1, 2], [6, 7]]), np.array([1, 11])),
        ]:
            match_idxs = mu.match_template(x=x, seq=seq)
            assert np.sum((np.array(match_idxs) - exp_matches)**2) < 0.0000000001, \
                'Match indexes ' + str(match_idxs) + ' not ' + str(exp_matches)
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    MathUtilsUnitTest(logger=lgr).test()
    mu = MathUtils(logger=lgr)
    res = mu.sample_random_no_repeat(
        list = np.arange(100).tolist() + np.arange(100).tolist(),
        n = 100,
    )
    res.sort()
    print(res)
    exit(0)
