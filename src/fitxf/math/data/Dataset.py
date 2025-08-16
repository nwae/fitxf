import logging
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from fitxf.utils import Logging, Env, Pandas


class DatasetUtil:

    def __init__(
            self,
            cache_dir: str | None = None,
            logger: logging.Logger | None = None,
    ):
        self.cache_dir = cache_dir
        self.logger = logger if logger is not None else logging.getLogger()
        self.dataset_dict = None
        self.dataset_path, self.dataset_name = None, None
        return

    def create_dataset_from_pandas(
            self,
            df_train: pd.DataFrame,
            df_dev: pd.DataFrame | None = None,
            df_test: pd.DataFrame | None = None,
    ):
        tmp_dict = {}
        for k, df in [('train', df_train), ('dev', df_dev), ('test', df_test)]:
            if df is not None:
                tmp_dict[k] = df
        self.dataset_dict = DatasetDict(tmp_dict)
        return self.dataset_dict

    def download(
            self,
            # e.g. 'sentence-transformers/all-nli
            dataset_path: str,
            # e.g. triplet
            dataset_name: str,
    ):
        if len(dataset_name) > 0:
            self.dataset_dict = load_dataset(
                path = dataset_path,
                name = dataset_name,
                # data_dir = self.cache_dir,
            )
        else:
            self.dataset_dict = load_dataset(path=dataset_path)
        self.logger.info(
            'Dataset path "' + str(dataset_path) + '", name "' + str(dataset_name)
            + '" downloaded, dataset type "' + str(type(self.dataset_dict)) + '", keys ' + str(self.get_dataset_keys())
            + ', length ' + str(len(self.dataset_dict)) + ', sample from key "' + str(self.get_dataset_keys()[0])
            + '": ' + str(self.dataset_dict[self.get_dataset_keys()[0]][0:3])
        )
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        if os.path.exists(self.cache_dir):
            dir = self.cache_dir + '/' + str(dataset_path) + '/' + str(dataset_name)
            os.makedirs(name=dir)
            for key in self.dataset_dict.keys():
                df = self.get_data(dataset_key=key)
                save_filepath = dir + '/' + str(key) + '.csv'
                df.to_csv(path_or_buf=save_filepath)
                self.logger.info('Saved dataset key "' + str(key) + '" to file "' + str(save_filepath) + '"')

        return

    def get_dataset_keys(self):
        return list(self.dataset_dict.keys())

    def is_dataset_exist(
            self,
            dataset_key: str,
    ):
        return dataset_key in self.dataset_dict.keys()

    def get_data(
            self,
            dataset_key: str,
            # <=0 means select all
            select_range: int = 0,
            # allowed values, "", "pandas
            return_type: str = "",
    ) -> Dataset | pd.DataFrame | None:
        assert type(self.dataset_dict) in [DatasetDict]

        if not self.is_dataset_exist(dataset_key=dataset_key):
            self.logger.warning(
                'Dataset key "' + str(dataset_key) + '" not found in dataset "' + str(self.dataset_path)
                + '-' + str(self.dataset_name) + '"'
            )
            return None

        if select_range > 0:
            data = self.dataset_dict[dataset_key].select(range(0, select_range))
        else:
            data = self.dataset_dict[dataset_key]

        self.logger.info(
            'Data type "' + str(type(data)) + ' for dataset key "' + str(dataset_key) + '", length ' + str(len(data))
        )
        if return_type == 'pandas':
            df = data.to_pandas()
            self.logger.info(
                'Converted to pandas dataframe: ' + str(df)
            )
            return df
        else:
            return data


if __name__ == '__main__':
    Pandas.increase_display()
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    er = Env(logger=lgr)
    ds = DatasetUtil(
        cache_dir = er.REPO_DIR + './data/datasets/latest',
        logger = lgr,
    )
    ds.download(
        dataset_path = 'mteb/banking77',
        # dataset_name = 'triplet',
        dataset_name = 'default'
    )
    df = ds.get_data(
        dataset_key = "train",
        select_range = 2,
        return_type = 'pandas',
    )
    lgr.info('Data in pandas:\n' + str(df))

    ds2 = DatasetUtil(logger=lgr)
    ds2.create_dataset_from_pandas(df_train=df)
    lgr.info('Dataset from pandas:\n' + str(ds2.dataset_dict))
    exit(0)
