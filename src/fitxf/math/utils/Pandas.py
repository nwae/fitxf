import logging
import pandas as pd
import numpy as np
from fitxf.math.utils.Logging import Logging


class Pandas:

    @staticmethod
    def increase_display(
            display_max_rows = 500,
            display_max_cols = 500,
            display_width = 1000,
    ):
        pd.set_option('display.max_rows', display_max_rows)
        pd.set_option('display.max_columns', display_max_cols)
        pd.set_option('display.width', display_width)
        return

    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    #
    # The problem with functions like numpy.unique() is it assumes super clean data,
    # thus if a column has mixed str/int/nan/etc, it won't work.
    # So we implement something that handles all that dirt.
    #
    def get_unique_segments(
            self,
            df: pd.DataFrame,
            # we will do conversion to the correct types if given
            column_types: list = None,
            # allowed values "numpy", ""
            method: str = ''
    ):
        if column_types is not None:
            assert len(column_types) == len(df.columns)
        tmp_groups = df.to_records(index=False)
        if method == "numpy":
            unique_groups = np.unique(tmp_groups)
        else:
            self.logger.debug('Existing groups: ' + str(tmp_groups))
            unique_groups = {}
            for grp in tmp_groups:
                grp_key = "\t".join([str(v) for v in grp])
                if grp_key not in unique_groups.keys():
                    if column_types is not None:
                        unique_groups[grp_key] = [column_types[i](v) for i, v in enumerate(grp)]
                    else:
                        unique_groups[grp_key] = grp
                    self.logger.debug('Added group ' + str(grp))

        return unique_groups

    def group_by_multi_agg(
            self,
            df: pd.DataFrame,
            cols_groupby: list,
            # e.g. {'sales': ['sum'], 'quality': ['mean', 'median']}
            agg_dict: dict,
    ):
        df_agg = df.groupby(
            by = cols_groupby,
            as_index = False,
        ).agg(agg_dict)
        # rename columns for user
        cols_renamed = list(cols_groupby)
        for col, aggregations in agg_dict.items():
            for ag in aggregations:
                cols_renamed.append(str(col) + '_' + str(ag))
        df_agg.columns = cols_renamed
        return df_agg


class PandasUnitTest:
    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        pd_utils = Pandas(logger=lgr)
        columns = ['date', 'count', 'label']
        col_types = [str, int, str]
        records = [
            ('2025-04-10',   2, 'marketing'),
            ('2025-04-10',   2, 'research'),
            ('2025-04-10', '2', 'research'),
            ('2025-04-10',   2, 'research'),
            ('2025-04-10',   2, 'research'),
        ]
        exp_sgmts = {
            '2025-04-10\t2\tmarketing': ['2025-04-10', 2, 'marketing'],
            '2025-04-10\t2\tresearch': ['2025-04-10', 2, 'research'],
        }
        unique_groups = pd_utils.get_unique_segments(
            df = pd.DataFrame.from_records(records, columns=columns),
            column_types = col_types,
        )
        self.logger.info(unique_groups)
        assert unique_groups == exp_sgmts, 'Got unique groups\n' + str(unique_groups) + '\nnot\n' + str(exp_sgmts)

        df_test = pd.DataFrame.from_records([
            {'shop': 'A', 'prd': 'bread',  'qnty': 120, 'qlty': 0.8},
            {'shop': 'A', 'prd': 'bread',  'qnty': 200, 'qlty': 0.7},
            {'shop': 'A', 'prd': 'bread',  'qnty': 80,  'qlty': 0.8},
            {'shop': 'B', 'prd': 'bread',  'qnty': 30,  'qlty': 0.5},
            {'shop': 'B', 'prd': 'bread',  'qnty': 20,  'qlty': 0.4},
            {'shop': 'B', 'prd': 'bread',  'qnty': 40,  'qlty': 0.2},
            {'shop': 'A', 'prd': 'butter', 'qnty': 40,  'qlty': 0.7},
            {'shop': 'A', 'prd': 'butter', 'qnty': 30,  'qlty': 0.7},
            {'shop': 'A', 'prd': 'butter', 'qnty': 50,  'qlty': 0.6},
        ])
        df_agg = pd_utils.group_by_multi_agg(
            df = df_test,
            cols_groupby = ['shop', 'prd'],
            agg_dict = {
                'qnty': ['sum'],
                'qlty': ['sum', 'mean', 'median'],
            }
        )
        self.logger.info('Aggregation:\n' + str(df_agg))
        assert list(df_agg.columns) == ['shop', 'prd', 'qnty_sum', 'qlty_sum', 'qlty_mean', 'qlty_median']
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    PandasUnitTest(logger=lgr).test()
    exit(0)
