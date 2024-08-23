import logging
import os
import numpy as np
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Profile import Profiling


# Important Note:
#     Logging of huge pandas/numpy arrays commented out in release version
#     as it slows down code by hundreds of milliseconds.
class MathUtils:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.enable_slow_logging_of_numpy = os.environ.get("ENABLE_SLOW_LOGGING", "false").lower() in ["true", "1"]
        return

    def match_template_1d(
            self,
            x: np.ndarray,
            seq: np.ndarray,
            # None: label on start positions of sequences found
            # "all": label on all positions
            label_positions = None,
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
        if self.enable_slow_logging_of_numpy:
            self.logger.debug('Template matching indices 2D structure: ' + str(template_matching_indices))
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
        if self.enable_slow_logging_of_numpy:
            self.logger.debug('Template matching indices final: ' + str(template_matching_indices))
        # Find matches
        template_matches = x[template_matching_indices] == seq
        if self.enable_slow_logging_of_numpy:
            self.logger.debug('Template matches for seq ' + str(seq) + ': ' + str(template_matches))

        #
        # nan means "*" match like in string regex
        #
        nan_positions = np.isnan(seq)
        if self.enable_slow_logging_of_numpy:
            self.logger.debug('nan positions: ' + str(nan_positions))
        template_matches = 1 * (template_matches | nan_positions)
        if self.enable_slow_logging_of_numpy:
            self.logger.debug('Template matches with nan for seq ' + str(seq) + ': ' + str(template_matches))

        # Match is when all are 1's
        match_start_indexes = 1 * (np.sum(template_matches, axis=-1) == len(seq))
        if self.enable_slow_logging_of_numpy:
            self.logger.debug('Match start indexes: ' + str(match_start_indexes))
        match_all_indexes = np.convolve(match_start_indexes, np.ones(l_seq, dtype=int), mode='full')
        if self.enable_slow_logging_of_numpy:
            self.logger.debug('Match all indexes  : ' + str(match_all_indexes))

        # Get the range of those indices as final output
        if match_start_indexes.any() > 0:
            res =  np.argwhere(match_start_indexes == 1).flatten().tolist()
            # return {
            #     'match_indexes': np.where(match_start_indexes == 1).flatten().tolist(),
            #     'match_sequence': np.where(np.convolve(match_start_indexes, np.ones((l_seq), dtype=int)) > 0)[0]
            # }
        else:
            res = []
            # return {
            #     'match_indexes': [],
            #     'match_sequence': [],  # No match found
            # }
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

        # Convert to ndim, same as converting to a base-N number
        if n_dim > 1:
            x_1d = x.flatten()
            seq_1d = seq.flatten()
            # Remove ending nan(s)
            for i in range(len(seq_1d)):
                if np.isnan(seq_1d[-1]):
                    seq_1d = seq_1d[:-1]
            if self.enable_slow_logging_of_numpy:
                self.logger.debug('Sequence flattened ' + str(seq_1d))

            match_start_indexes_1d = self.match_template_1d(x=x_1d, seq=seq_1d)
            if self.enable_slow_logging_of_numpy:
                self.logger.debug('Match 1d result ' + str(match_start_indexes_1d))

            bases = list(x.shape) + [1]
            converted_bases = []
            for idx in match_start_indexes_1d:
                nbr_rep = self.convert_to_multibase_number(n=idx, bases=bases, min_digits=x.ndim)
                converted_bases.append(nbr_rep)
                if self.enable_slow_logging_of_numpy:
                    self.logger.debug('Converted idx ' + str(idx) + ' to base ' + str(bases) + ' number: ' + str(nbr_rep))
            return converted_bases
        else:
            return self.match_template_1d(x=x, seq=seq)

    def convert_to_multibase_number(
            self,
            n: int,      # base 10 number
            bases: list,   # base to convert to, e.g. [6, 11, 1] --> last digit always 1
            min_digits: int = 0,
    ):
        assert n >= 0
        nbr_rep = []
        base = 1
        for idx in range(len(bases)-1):
            base *= bases[-(idx+2)]
            remainder = int(n % base)
            nbr_rep.append(remainder)
            n = (n - remainder) / base
            if self.enable_slow_logging_of_numpy:
                self.logger.debug('idx=' + str(idx) + ', base=' + str(base) + ', remainder=' + str(remainder))
        if n > 0:
            nbr_rep.append(n)
        while len(nbr_rep) < min_digits:
            nbr_rep.append(0)
        nbr_rep.reverse()
        return nbr_rep

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
        self.mu = MathUtils(logger=self.logger)
        return

    def test(self):
        for n, bases, exp in [
            (19, [5, 1], [3, 4]),
            (29, [13, 1], [2, 3]),
            (38, [13, 1], [2, 12]),
            (0, [3, 2, 1], [0, 0]),
            (1, [3, 2, 1], [0, 1]),
            (5, [3, 2, 1], [2, 1]),
            (5, [3, 2, 1, 1], [2, 1, 0]),
            (5, [3, 2, 1, 1, 1], [2, 1, 0, 0]),
            (29, [30, 20, 1], [1, 9]),
        ]:
            res = self.mu.convert_to_multibase_number(n=n, bases=bases, min_digits=len(bases)-1)
            assert res == exp, \
                'Test base convertion n=' + str(n) + ', bases=' + str(bases) + ', exp=' + str(exp) + ', res=' + str(res)

        self.test_1d()
        self.logger.info('1-DIMENSION TESTS PASSED')
        self.test_2d()
        self.logger.info('2-DIMENSION TESTS PASSED')

        os.environ["ENABLE_SLOW_LOGGING"] = "false"
        rps_no_logging = self.test_speed()

        os.environ["ENABLE_SLOW_LOGGING"] = "true"
        try:
            self.test_speed()
        except Exception as ex:
            self.logger.info(
                'Expected to fail when slow logging enabled (RPS no logging=' + str(rps_no_logging) + '): ' + str(ex)
            )
        self.logger.info('SPEED TESTS PASSED')

        self.logger.info('ALL TESTS PASSED')
        return

    def test_1d(self):
        # Test 1D
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        x = np.arange(20) % 10
        for seq, exp_matches in [
            (np.array([1, 2, 3, 4]), np.array([1, 11])),
            (np.array([1, np.nan, np.nan, 4]), np.array([1, 11])),
            (np.array([9, 10, 11]), np.array([9])),
            # (np.array([1, 3, 5]), []),
        ]:
            match_idxs = self.mu.match_template(x=x, seq=seq)
            assert np.sum((np.array(match_idxs) - exp_matches)**2) < 0.0000000001, \
                'Match indexes ' + str(match_idxs) + ' not ' + str(exp_matches)
        return

    def test_2d(self):
        # Test 2D
        # [[0 1 2 3 4]
        #  [5 6 7 8 9]
        #  [0 1 2 3 4]
        #  [5 6 7 8 9]]
        x = np.arange(20) % 10
        x.resize((4, 5))
        nan = np.nan
        self.logger.info('2D test data:\n' + str(x))

        for seq, exp_matches in [
            (np.array([[1, 2, nan, nan, nan], [6, 7, nan, nan, nan]]), np.array([[0,1], [2,1]])),
            # (np.array([[1, 2], [6, 7]]), np.array([[0, 1], [2, 1]])),
        ]:
            match_idxs = self.mu.match_template(x=x, seq=seq)
            assert np.sum((np.array(match_idxs) - exp_matches)**2) < 0.0000000001, \
                'Match indexes ' + str(match_idxs) + ' not ' + str(exp_matches)
        return

    def test_speed(self):
        new_obj = MathUtils(logger=self.logger)
        profiler = Profiling(logger=self.logger)
        x = np.arange(20) % 10
        x.resize((4, 5))
        nan = np.nan
        start_time = profiler.start()
        n = 1000
        for i in range(n):
            _ = new_obj.match_template(
                x = x,
                seq = np.array([[1, 2, nan, nan, nan], [6, 7, nan, nan, nan]]),
            )
        diffsecs = profiler.get_time_dif_secs(start=start_time, stop=profiler.stop())
        rps = round(n / diffsecs, 3)
        msec_avg = round(1000 * diffsecs / n, 3)
        self.logger.info(
            'RPS match template n=' + str(n) + ', total secs=' + str(diffsecs) + ', rps=' + str(rps)
            + ', msec avg=' + str(msec_avg)
        )
        assert rps > 10000, 'FAILED RPS n=' + str(n) + ', total=' + str(diffsecs) + 's, rps=' + str(rps)
        assert msec_avg < 0.1, 'FAILED RPS n=' + str(n) + ', total=' + str(diffsecs) + 's, msec avg=' + str(msec_avg)
        return rps


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    MathUtilsUnitTest(logger=lgr).test()
    mu = MathUtils(logger=lgr)
    res = mu.sample_random_no_repeat(
        list = np.arange(100).tolist() + np.arange(100).tolist(),
        n = 100,
    )
    res.sort()
    print(res)
    exit(0)
