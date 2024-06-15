import pandas as pd


class Pandas:

    @staticmethod
    def increase_display(
            display_max_rows = 500,
            display_max_cols = 500,
            display_width    = 1000,
    ):
        pd.set_option('display.max_rows', display_max_rows)
        pd.set_option('display.max_columns', display_max_cols)
        pd.set_option('display.width', display_width)