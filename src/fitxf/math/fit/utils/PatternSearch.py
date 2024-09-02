import logging
import traceback
import numpy as np
import pandas as pd
from fitxf import MathUtils
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Profile import Profiling
from fitxf.math.utils.Pandas import Pandas


class PatternSearch:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()

        self.profiler = Profiling(logger=self.logger)
        self.math_utils= MathUtils(logger=self.logger)
        return

    def heuristic_guess_start_search_locations(
            self,
            x: np.ndarray,
            possible_start_values: np.ndarray,
            seq_len: int,
            top_k: int = 10,
    ):
        assert seq_len <= len(x), 'Sequence length must be <= ' + str(len(x))
        start_time = self.profiler.start()

        # Find indexes with the possible start values
        test_diff = x[:,None] - possible_start_values
        # self.logger.debug('Test diff ' + str(test_diff))
        indexes_possible = np.argwhere(np.prod(test_diff, axis=-1) == 0).flatten()
        # Move one character to the next character, since these are separators
        indexes_possible = np.minimum(len(x)-1, indexes_possible + 1)
        # self.logger.debug(
        #     'Possible indexes ' + str(indexes_possible) + ' at ' + str([chr(v) for v in x[indexes_possible]])
        # )

        successful_indexes = []
        hint_prefix_indexes = {}
        for i in indexes_possible:
            if i + seq_len > len(x):
                continue
            if i in successful_indexes:
                continue
            seq_start, seq_end = i, i + seq_len
            seq_i = x[seq_start:seq_end]
            seq_i_str = ''.join([chr(v) for v in seq_i])

            match_indexes = self.math_utils.match_template(
                x = x,
                seq = seq_i,
                return_only_start_indexes = True,
            )
            success = len(match_indexes) >= 2
            if not success:
                continue
            else:
                hint_prefix_indexes[seq_i_str] = hint_prefix_indexes.get(seq_i_str, []) + match_indexes
                successful_indexes += match_indexes
                self.logger.debug(
                    'Match indexes at index ' + str(seq_start) + ' for seq len ' + str(seq_len) + ' "'
                    + str(''.join([chr(v) for v in seq_i])) + '": ' + str(match_indexes) + ', hint indexes now: '
                    + str(hint_prefix_indexes)
                )

        successful_indexes.sort()
        self.logger.info('Successful hint indexes: ' + str(successful_indexes))

        if len(hint_prefix_indexes) > 0:
            # sort and select top_k
            records = [
                {'prefix': prefix, 'indexes': indexes, 'count': len(indexes)}
                for prefix, indexes in hint_prefix_indexes.items()
            ]
            df_records = pd.DataFrame.from_records(records)
            df_records = df_records.sort_values(by=['count'], ascending=False)
            df_records = df_records[0:min(top_k, len(df_records))]
            self.logger.info('Final sorted hint prefix indexes: ' + str(df_records))
            hint_prefix_indexes_sorted = {r['prefix']: r['indexes'] for r in df_records.to_dict(orient='records')}
        else:
            hint_prefix_indexes_sorted = {}

        diff_msecs = round(1000 * self.profiler.get_time_dif_secs(start=start_time, stop=self.profiler.stop()), 2)
        self.logger.info('Hint indexes search for ' + str(possible_start_values) + ' took ' + str(diff_msecs) + 'ms')

        return hint_prefix_indexes_sorted

    def repeat_density_and_coverage(
            self,
            match_indexes: list,
            seq_len: int,
            total_len: int,
    ):
        i_start = np.min(match_indexes)
        i_end = min(np.max(match_indexes) + seq_len, total_len)
        # Density of repeat sequences in the range of valid repeats
        density = round(len(match_indexes) * seq_len / (i_end - i_start), 3)
        # Coverage of the repeat range in the whole length
        coverage = round((i_end - i_start) / total_len, 3)
        return density, coverage

    def find_repeat_sequences(
            self,
            x: np.ndarray,
            hint_separators: np.ndarray,
            density_repeat_thr: float = 0.8,
            coverage_thr: float = 0.5,
            # roughly 10 words
            min_seq_len: int = 64,
            top_k_hints: int = 10,
            stagnant_at_optimal_count_thr: int = 5,
            string_convert_result: bool = False,
    ):
        start_time = self.profiler.start()

        hint_prefix_indexes = self.heuristic_guess_start_search_locations(
            x = x,
            possible_start_values = hint_separators,
            seq_len = min_seq_len,
            top_k = top_k_hints,
        )

        repeat_indexes_and_texts = []
        for prefix, indexes in hint_prefix_indexes.items():
            match_indexes_start = np.array(indexes)
            intervals = match_indexes_start[1:] - match_indexes_start[:-1]
            max_interval = np.min(intervals)
            min_interval = min_seq_len
            if min_interval > max_interval:
                continue
            seq_list = [min_interval]
            self.logger.info(
                'Using min interval ' + str(min_seq_len) + ', max interval ' + str(max_interval)
                + ' for index intervals: ' + str(intervals)
            )
            ind_optimal = None
            break_reason = None
            stagnant_count = 0
            while True:
                # TODO Don't just take the first one
                # Take the first one, the last one might be too short and will mistakenly cut the match sequence
                idx_start = match_indexes_start[0]
                idx_end = min(idx_start + seq_list[-1], len(x))
                seq_i = x[idx_start:idx_end]
                self.logger.debug(
                    'Match template for idx [' + str(idx_start) + ',' + str(idx_end) + '], seq len ' + str(seq_list[-1])
                    + ' "' + str(''.join([chr(v) for v in seq_i])) + '"'
                )
                match_indexes_end_tmp = self.math_utils.match_template(
                    x = x,
                    seq = seq_i,
                    return_only_start_indexes = True,
                )
                self.logger.debug(
                    'Match indexes for seq len ' + str(seq_list[-1]) + ' "' + str(''.join([chr(v) for v in seq_i]))
                    + '": ' + str(match_indexes_end_tmp)
                )

                is_success_match_seq = len(match_indexes_end_tmp) >= 2
                improve_from_prev_optimal = False

                #
                # Check to see if we got new optimal if success in matching
                #
                if is_success_match_seq:
                    ind_tmp = self.get_repeat_indicators(
                        match_indexes = match_indexes_end_tmp,
                        seq_list = seq_list,
                        x = x,
                        string_convert = string_convert_result,
                    )
                    density, coverage, score = ind_tmp["density"], ind_tmp["coverage"], ind_tmp["score"]
                    density_cond_met = density >= density_repeat_thr
                    coverage_cond_met = coverage >= coverage_thr
                    prev_score = ind_optimal['score'] if ind_optimal is not None else -np.inf
                    improve_from_prev_optimal = (density_cond_met and coverage_cond_met) and (score > prev_score)

                    if improve_from_prev_optimal:
                        self.logger.info(
                            'Found new optimal from prev ' + str(ind_optimal) + '. Improved: ' + str(ind_tmp)
                        )
                        ind_optimal = ind_tmp
                    else:
                        tmp_prev_optimal = {k: v for k, v in ind_optimal.items() if k not in ["s_parts", "prefix"]} \
                            if ind_optimal is not None else None
                        self.logger.info(
                            'No improve from previous optimal ' + str(tmp_prev_optimal) + ' for: '
                            + str({k: v for k, v in ind_tmp.items() if k not in ["s_parts", "prefix"]})
                        )

                    # if (seq_list[-1] == max_interval):
                    #     self.logger.debug(
                    #         'Found optimal by early condition, max seq len ' + str(seq_list)
                    #         + ', max interval ' + str(max_interval)
                    #     )
                    #     break_reason = 'hit max interval'
                    #     break
                    if improve_from_prev_optimal:
                        self.logger.info(
                            'Found optimal by early condition, max seq len ' + str(seq_list)
                            + ', max interval ' + str(max_interval) + ', density ' + str(density)
                            + ' (thr=' + str(density_repeat_thr) + ', cond=' + str(density_cond_met)
                            + '), coverage ' + str(coverage) + '(thr=' + str(coverage_thr)
                            + ', cond=' + str(coverage_cond_met) + ')'
                        )
                        break_reason = 'density/coverage condition met'
                        break
                    elif stagnant_count >= stagnant_at_optimal_count_thr:
                        self.logger.info(
                            'Stagnant for last ' + str(stagnant_count) + ' tries >= '
                            + str(stagnant_at_optimal_count_thr)
                        )
                        break_reason = 'stagnant ' + str(stagnant_count)
                        break

                #
                # Adjust new max/min intervals first
                #
                if not is_success_match_seq:
                    # The last seq len had no match, means we know now the upper bound is at most 1 less
                    tmp = max_interval
                    max_interval = min(max_interval, seq_list[-1] - 1)
                    self.logger.debug(
                        'No match at seq len ' + str(seq_list[-1]) + ', max interval changed from ' + str(tmp)
                        + ' --> ' + str(max_interval)
                    )
                else:
                    # The last seq len search successful, means we know lower bound is at least seq len + 1
                    tmp = min_interval
                    # if improve_from_prev_optimal:
                    #     # Be more daring, move lower bound up by at least half the interval
                    #     min_interval = max(min_interval, int((min_interval + max_interval) / 2))
                    # else:
                    # No change or just add 1 to lower bound, bcos was not optimal
                    min_interval = min_interval+1 if seq_list[-1]==min_interval else min_interval
                    self.logger.debug(
                        'Have match at seq len ' + str(seq_list[-1])
                        + ', min interval changed (or possibly remained) from ' + str(tmp)
                        + ' --> ' + str(min_interval) + ', upper bound ' + str(max_interval)
                        + '. Improve from prev optimal ' + str(improve_from_prev_optimal)
                    )

                remaining_indexes_to_try = np.arange(max_interval - min_interval + 1) + min_interval
                remaining_indexes_to_try = list(
                    set(remaining_indexes_to_try.tolist()).difference(set(seq_list))
                )
                self.logger.debug('Remaining indexes to try: ' + str(remaining_indexes_to_try))
                if not remaining_indexes_to_try:
                    self.logger.debug(
                        'Cannot decrease/increase seq len anymore, min interval=' + str(min_interval)
                        + ', max=' + str(max_interval) + '. No more remaining indexes available in history '
                        + str(seq_list)
                    )
                    break_reason = 'no more valid indexes to try'
                    break

                #
                # Adjust next move
                #
                if not is_success_match_seq:
                    stagnant_count = 0
                    # no match, decrease seq len by approx. half
                    seq_len_new = remaining_indexes_to_try[int(len(remaining_indexes_to_try) / 2)]
                    self.logger.info(
                        'No success, moved seq len from ' + str(seq_list[-1]) + ' --> ' + str(seq_list)
                        + ', lower/upper bound [' + str(min_interval) + ', ' + str(max_interval) + ']'
                    )
                    seq_list.append(seq_len_new)
                elif not improve_from_prev_optimal:
                    stagnant_count += 1
                    seq_len_new = np.random.choice(remaining_indexes_to_try)
                    self.logger.info(
                        'Did not improve optimal, moved seq len randomly from ' + str(seq_list[-1])
                        + ' --> ' + str(seq_len_new)
                        + ', lower/upper bound [' + str(min_interval) + ', ' + str(max_interval) + ']'
                    )
                    seq_list.append(seq_len_new)
                else:
                    stagnant_count = 0
                    # positive match, increase seq len
                    seq_len_new = np.max(remaining_indexes_to_try)
                    self.logger.info(
                        'Success with optimal, moved seq len from ' + str(seq_list[-1]) + ' --> ' + str(seq_len_new)
                    )
                    seq_list.append(seq_len_new)

            if ind_optimal is not None:
                ind_optimal['break_reason'] = break_reason
                repeat_indexes_and_texts.append(ind_optimal)
                self.logger.debug('Appended: ' + str(ind_optimal))
        try:
            df_repeat_indexes = pd.DataFrame.from_records(repeat_indexes_and_texts)
            df_repeat_indexes = df_repeat_indexes.sort_values(by=['score'], ascending=False)
            self.logger.debug('Repeat indexes: ' + str(df_repeat_indexes))
            final_matches = df_repeat_indexes.to_dict(orient='records')
        except Exception as ex:
            self.logger.debug('Possibly no matches found: ' + str(ex) + ' Stack trace: ' + str(traceback.format_exc()))
            final_matches = None

        diff_msecs = round(1000 * self.profiler.get_time_dif_secs(start=start_time, stop=self.profiler.stop()), 2)
        self.logger.info(
            'Repeat sequence search took ' + str(diff_msecs) + 'ms. Sequence lengths in iteration: '
        )
        return final_matches

    def get_repeat_indicators(
            self,
            match_indexes: list,
            seq_list: list,
            x: np.ndarray,
            string_convert: bool = False,
    ):
        if string_convert:
            string = ''.join([chr(v) for v in x])
        else:
            string = x
        # self.logger.debug('Match indexes end: ' + str(match_indexes))
        # self.logger.debug('Match indexes end: ' + str([string[i:(i + 50)] for i in match_indexes]))
        seq_len = seq_list[-1]
        i_start = np.min(match_indexes)
        i_end = min(np.max(match_indexes) + seq_len, len(x))
        s_parts = [string[:i_start], string[i_start:i_end], string[i_end:]]
        prefix = string[i_start:(i_start+seq_len)]
        # self.logger.debug('New prefix "' + str(prefix) + '"')

        density, coverage = self.repeat_density_and_coverage(
            match_indexes = match_indexes,
            seq_len = seq_len,
            total_len = len(string)
        )
        # self.logger.debug(
        #     'Seq len ' + str(seq_len) + ' "' + str(string[i_start:(i_start + seq_len)])
        #     + '", start ' + str(i_start) + ', end ' + str(i_end)
        #     + ', density of repeats ' + str(density) + ', coverage ' + str(coverage)
        # )

        tmp = np.array(match_indexes)
        intervals = tmp[1:] - tmp[:-1]
        return {
            'indexes': match_indexes,
            'seq_len': seq_len,
            'seq_list': seq_list,
            'prefix': prefix,
            'index_start': i_start,
            'index_end': i_end,
            'min_interval': np.min(intervals),
            'max_interval': np.max(intervals),
            # 'texts': [string[i:(i+seq_len_end)] for i in match_indexes_end],
            'density': density,
            'coverage': coverage,
            'score': density * coverage,
            's_parts': s_parts,
        }


if __name__ == '__main__':
    Pandas.increase_display()
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    ps = PatternSearch(logger=lgr)
    res = ps.find_repeat_sequences(
        x = np.array([
            ord(c) for c in "pattern 1 pattern 2 pattern 3 pattern 4 ..."]
        ),
        min_seq_len = 4,
        hint_separators = np.array([ord(c) for c in ["1", " "]]),
        string_convert_result = True,
        density_repeat_thr = 0.5,
        coverage_thr = 0.5,
    )
    print(res)
    exit(0)
