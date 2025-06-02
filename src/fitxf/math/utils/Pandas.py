import logging
import pandas as pd
import numpy as np
from fitxf.math.utils.Logging import Logging


class Pandas:

    CUSTOM_AGGREGATIONS = ('weighted_mean',)

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
            # for custom weighted aggregations
            weights: list | None = None,
            rename_with_agg: bool = False,
    ) -> pd.DataFrame:
        # Rearrange and check for custom aggregation like "weighted_mean"
        agg_dict_standard = {}
        agg_dict_custom = {}
        for k, aggs in agg_dict.items():
            have_custom = len(set(aggs).intersection(self.CUSTOM_AGGREGATIONS)) > 0
            if have_custom:
                agg_dict_custom[k] = [v for v in aggs if v in self.CUSTOM_AGGREGATIONS]
                agg_dict_standard[k] = [v for v in aggs if v not in self.CUSTOM_AGGREGATIONS]
            else:
                agg_dict_standard[k] = aggs
        self.logger.debug(
            'Split up aggregation dict ' + str(agg_dict) + ' to standard aggregations ' + str(agg_dict_standard)
            + ' and custom aggregations ' + str(agg_dict_custom)
        )
        df_agg_std = self.group_by_multi_agg_standard(
            df = df,
            cols_groupby = cols_groupby,
            agg_dict = agg_dict_standard,
            rename_with_agg = rename_with_agg,
        )
        df_agg_custom = self.group_by_multi_agg_custom(
            df = df,
            cols_groupby = cols_groupby,
            agg_dict = agg_dict_custom,
            weights = weights,
            rename_with_agg = rename_with_agg,
        )
        if df_agg_custom is not None:
            df_agg_final = df_agg_std.merge(
                df_agg_custom,
                how = 'left',
                left_on = cols_groupby,
                right_on = cols_groupby
            )
        else:
            df_agg_final = df_agg_std
        return df_agg_final

    def group_by_multi_agg_custom(
            self,
            df: pd.DataFrame,
            cols_groupby: list,
            # e.g. {'sales': ['sum'], 'quality': ['mean', 'median']}
            agg_dict: dict,
            weights: list,
            rename_with_agg: bool = False,
    ) -> pd.DataFrame:
        df_agg = None
        # do 1 by 1 for custom aggregations
        for col, aggregations in agg_dict.items():
            for custom_agg in aggregations:
                if custom_agg == 'weighted_mean':
                    col_name_agg = str(col) + '_weighted'
                    # don't modify original dataframe
                    df_copy = df[cols_groupby].copy()
                    df_copy['__weights'] = weights
                    df_copy[col_name_agg] = df_copy['__weights'] * df[col]
                    df_agg_tmp = df_copy.groupby(
                        by = cols_groupby,
                        as_index = False,
                    ).sum()
                    df_agg_tmp[col_name_agg] = df_agg_tmp[col_name_agg] / df_agg_tmp['__weights']
                    cols_keep = cols_groupby + [col_name_agg]
                    if df_agg is None:
                        df_agg = df_agg_tmp[cols_keep]
                    else:
                        df_agg = df_agg.merge(
                            df_agg_tmp[cols_keep],
                            how = 'left',
                            left_on = cols_groupby,
                            right_on = cols_groupby,
                        )
                    self.logger.debug(
                        'Aggregation by ' + str(custom_agg) + ' for column "' + str(col) + '":'
                        + str(df_agg)
                    )
                else:
                    raise Exception('Not supported custom aggregation "' + str(custom_agg) + '"')
        # self.logger.debug(df_agg)
        return df_agg

    def group_by_multi_agg_standard(
            self,
            df: pd.DataFrame,
            cols_groupby: list,
            # e.g. {'sales': ['sum'], 'quality': ['mean', 'median']}
            agg_dict: dict,
            rename_with_agg: bool = False,
    ) -> pd.DataFrame:
        # e.g. {'sales': ['sum'], 'quality': ['mean', 'median']}
        df_agg = df.groupby(
            by = cols_groupby,
            as_index = False,
        ).agg(agg_dict)

        # rename columns for user
        cols_renamed = list(cols_groupby)
        for col, aggregations in agg_dict.items():
            for ag in aggregations:
                cols_renamed.append(str(col) + '_' + str(ag) if rename_with_agg else str(col))
        df_agg.columns = cols_renamed
        return df_agg


class PandasUnitTest:
    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        Pandas.increase_display()
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
        exp_quantity_sum = [400, 120, 90]
        exp_quality_median = [0.8, 0.7, 0.4]
        exp_quality_mean = [0.76666667, 0.66666667, 0.36666667]
        exp_quality_weighted = [0.75, 0.65833333, 0.34444444]
        exp_quantity_weigted = [152.0,  41.66666667, 32.22222222]

        df_agg = pd_utils.group_by_multi_agg(
            df = df_test,
            cols_groupby = ['shop', 'prd'],
            agg_dict = {
                'qnty': ['sum', 'weighted_mean'],
                'qlty': ['sum', 'mean', 'median', 'weighted_mean'],
            },
            weights = list(df_test['qnty']),
            rename_with_agg = True,
        )
        df_agg = df_agg.sort_values(by=['shop', 'prd'], ascending=True)

        self.logger.info('Aggregation:\n' + str(df_agg))
        assert list(df_agg.columns) == [
            'shop', 'prd', 'qnty_sum', 'qlty_sum', 'qlty_mean', 'qlty_median',
            'qnty_weighted', 'qlty_weighted',
        ]
        assert list(df_agg['qnty_sum']) == exp_quantity_sum
        assert np.sum((np.array(df_agg['qlty_median']) - np.array(exp_quality_median))**2) < 0.0000000001, \
            'Got quality median ' + str(list(df_agg['qlty_median'])) + ' not ' + str(exp_quality_median)
        assert np.sum((np.array(df_agg['qlty_mean']) - np.array(exp_quality_mean))**2) < 0.0000000001, \
            'Got quality mean ' + str(list(df_agg['qlty_mean'])) + ' not ' + str(exp_quality_mean)
        assert np.sum((np.array(df_agg['qlty_weighted']) - np.array(exp_quality_weighted))**2) < 0.0000000001, \
            'Got quality weighted ' + str(list(df_agg['qlty_weighted'])) + ' not ' + str(exp_quality_weighted)
        assert np.sum((np.array(df_agg['qnty_weighted']) - np.array(exp_quantity_weigted))**2) < 0.0000000001, \
            'Got quantity weighted ' + str(list(df_agg['qnty_weighted'])) + ' not ' + str(exp_quantity_weigted)
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    PandasUnitTest(logger=lgr).test()
    exit(0)
