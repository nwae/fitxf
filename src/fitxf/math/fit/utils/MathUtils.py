import logging
import os
import platform
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
    ) -> dict:
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
            # res =  np.argwhere(match_start_indexes == 1).flatten().tolist()
            res = {
                'match_indexes': np.argwhere(
                    match_start_indexes == 1
                ).flatten().tolist(),
                'match_sequence': np.argwhere(
                    np.convolve(match_start_indexes, np.ones((l_seq), dtype=int) > 0)
                ).flatten().tolist()
            }
        else:
            # res = []
            res = {
                'match_indexes': [],
                'match_sequence': [],  # No match found
            }
        return res

    def match_template(
            self,
            x: np.ndarray,
            seq: np.ndarray,
            seq_real_shape: list = None,
            return_only_start_indexes = True,
    ) -> dict:
        x = np.array(x) if type(x) in (list, tuple) else x
        seq = np.array(seq) if type(seq) in (list, tuple) else seq
        assert x.ndim == seq.ndim, 'Dimensions do not match, x dim ' + str(x.ndim) + ', seq dim ' + str(seq.ndim)
        n_dim = x.ndim

        # Convert to ndim, same as converting to a base-N number
        if n_dim > 1:
            assert seq_real_shape is not None
            x_1d = x.flatten()
            seq_1d = seq.flatten()
            # count_valid = np.sum(1 * np.logical_not(np.isnan(seq_1d)))
            # Remove ending nan(s)
            for i in range(len(seq_1d)):
                if np.isnan(seq_1d[-1]):
                    seq_1d = seq_1d[:-1]
            # if self.enable_slow_logging_of_numpy:
            #     self.logger.debug('Sequence flattened ' + str(seq_1d))

            res = self.match_template_1d(x=x_1d, seq=seq_1d)
            match_start_indexes_1d, match_seq_1d = res['match_indexes'], res['match_sequence']
            if self.enable_slow_logging_of_numpy:
                self.logger.debug('Match 1d result ' + str(res))

            bases = list(x.shape) + [1]
            match_indexes = []
            match_sequence = []
            len_cycle = int(len(match_seq_1d) / len(match_start_indexes_1d))
            if self.enable_slow_logging_of_numpy:
                self.logger.debug(
                    'Length of cycle ' + str(len_cycle) + ' for match start indexes 1d ' + str(match_start_indexes_1d)
                    + ', match seq 1d: ' + str(match_seq_1d)
                )

            for i, idx_1d in enumerate(match_start_indexes_1d):
                nbr_rep = self.convert_to_multibase_number(n=idx_1d, bases=bases, min_digits=x.ndim)
                border_points = np.array(nbr_rep) + np.array(seq_real_shape) - 1
                check_if_overflow_x = np.min(np.array(x.shape) - 1 - border_points) < 0
                if check_if_overflow_x:
                    if self.enable_slow_logging_of_numpy:
                        self.logger.debug(
                            'Start index IGNORE ' + str(nbr_rep) + ', seq: ' + str(seq)
                            + ', shape real ' + str(seq_real_shape) + ', border points ' + str(border_points)
                            + ', check overflow ' + str(check_if_overflow_x)
                        )
                    continue
                # else:
                #     if self.enable_slow_logging_of_numpy:
                #         self.logger.debug(
                #             'Start index OK ' + str(nbr_rep) + ', seq: ' + str(seq)
                #             + ', shape real ' + str(seq_real_shape) + ', border points ' + str(border_points)
                #             + ', check overflow ' + str(check_if_overflow_x)
                #         )

                if self.enable_slow_logging_of_numpy:
                    self.logger.debug(
                        'Match indexes converted idx ' + str(idx_1d) + ' to base ' + str(bases) + ' number: ' + str(nbr_rep)
                    )
                i_start = i*len_cycle
                match_seq_1d_1cycle = match_seq_1d[i_start:(i_start + len_cycle)]
                if self.enable_slow_logging_of_numpy:
                    self.logger.debug(
                        'Match seq 1d 1 cycle at index ' + str(i_start) + ': ' + str(match_seq_1d_1cycle)
                    )
                coor = self.get_coordinates(
                    x = x,
                    match_seq_1d_1cycle = match_seq_1d_1cycle,
                    seq_original = seq,
                )
                # if is_valid:
                match_indexes.append(nbr_rep)
                match_sequence = match_sequence + coor
                # else:
                #     if self.enable_slow_logging_of_numpy:
                #         self.logger.info(
                #             'Invalid match sequence ' + str(match_seq_1d_1cycle) + ', coordinates ' + str(coor)
                #         )
            return match_indexes if return_only_start_indexes else \
                {'match_indexes': match_indexes, 'match_sequence': match_sequence}
        else:
            res = self.match_template_1d(x=x, seq=seq)
            return res['match_indexes'] if return_only_start_indexes else res

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
            # if self.enable_slow_logging_of_numpy:
            #     self.logger.debug('idx=' + str(idx) + ', base=' + str(base) + ', remainder=' + str(remainder))
        if n > 0:
            nbr_rep.append(n)
        while len(nbr_rep) < min_digits:
            nbr_rep.append(0)
        nbr_rep.reverse()
        return nbr_rep

    def get_coordinates(
            self,
            x: np.ndarray,
            match_seq_1d_1cycle: list,
            seq_original: np.ndarray,
    ) -> (list, bool):
        assert len(match_seq_1d_1cycle) > 0
        bases = list(x.shape) + [1]
        coordinates = []
        for idx in match_seq_1d_1cycle:
            nbr_rep = self.convert_to_multibase_number(n=idx, bases=bases, min_digits=x.ndim)
            coordinates.append(nbr_rep)
            if self.enable_slow_logging_of_numpy:
                self.logger.debug(
                    'Match sequence converted idx ' + str(idx) + ' to base ' + str(bases) + ' number: ' + str(nbr_rep)
                )
        if self.enable_slow_logging_of_numpy:
            self.logger.debug('Converted match seq ' + str(match_seq_1d_1cycle) + ' to ' + str(coordinates))
        # shape_len_1d = len(match_seq_1d_1cycle)
        # self.logger.info('Shape len 1d ' + str(shape_len_1d) + ' for ' + str(seq))
        # Check if result spans across row/column/etc borders, which is not valid
        # span_min = np.min(np.array(coordinates[:shape_len_1d]), axis=0)
        # span_max = np.max(np.array(coordinates[:shape_len_1d]), axis=0)
        # span_shape = span_max - span_min + 1
        # seq_shape = np.array(list(seq_original.shape))
        # self.logger.info('Span min: ' + str(span_min) + ' for ' + str(match_sequence[:shape_len_1d]))
        # self.logger.info('Span max: ' + str(span_max) + ' for ' + str(match_sequence[:shape_len_1d]))
        # self.logger.info(
        #     'Span shape ' + str(span_shape) + ', seq shape ' + str(seq_shape) + ', diff ' + str(span_shape - seq_shape)
        # )
        # not_cross_rows = np.max(span_shape - seq_shape) <= 0
        # valid = not_cross_rows
        return coordinates

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
        for i, (seq, exp_matches, exp_seq) in enumerate([
            (np.array([1, 2, 3, 4]), np.array([1, 11]), np.array([1,  2,  3,  4, 11, 12, 13, 14])),
            (np.array([1, np.nan, np.nan, 4]), np.array([1, 11]), np.array([1,  2,  3,  4, 11, 12, 13, 14])),
            (np.array([9, 10, 11]), np.array([]), np.array([])),
            # (np.array([1, 3, 5]), []),
        ]):
            res = self.mu.match_template(
                x = x,
                seq = seq,
                return_only_start_indexes = False,
            )
            match_idxs, match_seq = res['match_indexes'], res['match_sequence']
            self.logger.info('Test result 1D #' + str(i) + ': ' + str(res))
            assert np.sum((np.array(match_idxs) - exp_matches)**2) < 0.0000000001, \
                '#' + str(i) + ' Match indexes ' + str(match_idxs) + ' not ' + str(exp_matches)
            assert np.sum((np.array(match_seq) - exp_seq)**2) < 0.0000000001, \
                '#' + str(i) +' Match sequence ' + str(match_seq) + ' not ' + str(exp_seq)
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

        for i, (seq, seq_ori_shape, exp_matches, exp_seq) in enumerate([
            (np.array([[1, 2, nan, nan, nan], [6, 7, nan, nan, nan]]), (2, 2), np.array([[0,1], [2,1]]),
             np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [2, 4], [3, 0], [3, 1], [3, 2]])),
            # (np.array([[1, 2], [6, 7]]), np.array([[0, 1], [2, 1]]), np.array([])),
            # Test invalid of cross rows
            (np.array([[4, 5, nan, nan, nan], [9, 0, nan, nan, nan]]), (2, 2), np.array([]),
             np.array([])),
        ]):
            res = self.mu.match_template(
                x = x,
                seq = seq,
                seq_real_shape = seq_ori_shape,
                return_only_start_indexes = False,
            )
            match_idxs, match_seq = res['match_indexes'], res['match_sequence']
            self.logger.info('Test result 2D #' + str(i) + ': ' + str(res))
            assert np.sum((np.array(match_idxs) - exp_matches)**2) < 0.0000000001, \
                '#' + str(i) + ' Match indexes ' + str(match_idxs) + ' not ' + str(exp_matches)
            assert np.sum((np.array(match_seq) - exp_seq) ** 2) < 0.0000000001, \
                '#' + str(i) + ' Match sequence ' + str(match_seq) + ' not ' + str(exp_seq)
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
                seq_real_shape = [2, 2],
                return_only_start_indexes = False,
            )
        diffsecs = profiler.get_time_dif_secs(start=start_time, stop=profiler.stop())
        rps = round(n / diffsecs, 3)
        msec_avg = round(1000 * diffsecs / n, 3)
        self.logger.info(
            'RPS match template n=' + str(n) + ', total secs=' + str(diffsecs) + ', rps=' + str(rps)
            + ', msec avg=' + str(msec_avg)
        )
        pf = platform.platform()
        pf_info = pf.split("-")
        processor = pf_info[2]
        if processor in ['x86_64']:
            rps_thr, msec_thr = 1000, 0.5
        else:
            rps_thr, msec_thr = 10000, 0.1
        self.logger.info(
            'Platform processor "' + str(processor) + '", "' + str(pf) + '": ' + str(pf_info)
            + ', rps thr ' + str(rps_thr)
        )

        assert rps > rps_thr, \
            'FAILED RPS n=' + str(n) + ', total=' + str(diffsecs) + 's, rps=' + str(rps)
        assert msec_avg < msec_thr, \
            'FAILED RPS n=' + str(n) + ', total=' + str(diffsecs) + 's, msec avg=' + str(msec_avg)
        return rps


if __name__ == '__main__':
    # os.environ["ENABLE_SLOW_LOGGING"] = "true"
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
