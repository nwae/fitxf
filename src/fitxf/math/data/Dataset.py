import logging
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from fitxf.utils import Logging, Env, Pandas


class DatasetUtil:

    def __init__(
            self,
            cache_dir: str | None = None,
            logger: Logging | None = None,
    ):
        self.cache_dir = cache_dir
        self.logger = logger if logger is not None else logging.getLogger()
        self.dataset = None
        self.dataset_path, self.dataset_name = None, None
        return

    def download(
            self,
            # e.g. 'sentence-transformers/all-nli
            dataset_path: str,
            # e.g. triplet
            dataset_name: str,
    ):
        if len(dataset_name) > 0:
            self.dataset = load_dataset(
                path = dataset_path,
                name = dataset_name,
                # data_dir = self.cache_dir,
            )
        else:
            self.dataset = load_dataset(path=dataset_path)
        self.logger.info(
            'Dataset path "' + str(dataset_path) + '", name "' + str(dataset_name)
            + '" downloaded, dataset type "' + str(type(self.dataset)) + '", keys ' + str(self.get_dataset_keys())
            + ', length ' + str(len(self.dataset)) + ', sample from key "' + str(self.get_dataset_keys()[0])
            + '": ' + str(self.dataset[self.get_dataset_keys()[0]][0:3])
        )
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        return

    def get_dataset_keys(self):
        return list(self.dataset.keys())

    def is_dataset_exist(
            self,
            dataset_key: str,
    ):
        return dataset_key in self.dataset.keys()

    def get_data(
            self,
            dataset_key: str,
            # <=0 means select all
            select_range: int = 0,
            # allowed values, "", "pandas
            return_type: str = "",
    ) -> Dataset | pd.DataFrame | None:
        assert type(self.dataset) in [DatasetDict]

        if not self.is_dataset_exist(dataset_key=dataset_key):
            self.logger.warning(
                'Dataset key "' + str(dataset_key) + '" not found in dataset "' + str(self.dataset_path)
                + '-' + str(self.dataset_name) + '"'
            )
            return None

        if select_range > 0:
            data = self.dataset[dataset_key].select(range(0, select_range))
        else:
            data = self.dataset[dataset_key]

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
        cache_dir = er.DATASET_DIR,
        logger = lgr,
    )
    ds.download(
        dataset_path = 'sentence-transformers/all-nli',
        dataset_name = 'triplet',
    )
    ds.get_data(
        dataset_key = "train",
        select_range = 2,
        return_type = 'pandas',
    )
    exit(0)
