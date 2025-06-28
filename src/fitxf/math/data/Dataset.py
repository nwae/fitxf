import logging
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from fitxf.utils import Logging, Env, Pandas


class DatasetUtil:

    def __init__(
            self,
            cache_dir: str,
            logger: Logging | None = None,
    ):
        self.cache_dir = cache_dir
        self.logger = logger if logger is not None else logging.getLogger()
        self.dataset = None
        return

    def download(
            self,
            # e.g. 'sentence-transformers/all-nli
            dataset_path: str,
            # e.g. triplet
            dataset_name: str,
    ):
        self.dataset = load_dataset(
            path = dataset_path,
            name = dataset_name,
            # data_dir = self.cache_dir,
        )
        self.logger.info(
            'Dataset path "' + str(dataset_path) + '", name "' + str(dataset_name)
            + '" downloaded, dataset type "' + str(type(self.dataset)) + '", keys ' + str(self.get_dataset_keys())
            + ', length ' + str(len(self.dataset)) + ', sample from key "' + str(self.get_dataset_keys()[0])
            + '": ' + str(self.dataset[self.get_dataset_keys()[0]][0:3])
        )
        return

    def get_dataset_keys(self):
        return list(self.dataset.keys())

    def get_data(
            self,
            dataset_key: str,
            return_type: str = 'dataframe',
    ) -> Dataset | pd.DataFrame:
        assert type(self.dataset) in [DatasetDict]
        data = self.dataset[dataset_key]
        df = data.to_pandas()
        self.logger.info(
            'Data type "' + str(type(data)) + ' for dataset key "' + str(dataset_key) + '", length ' + str(len(data))
            + ', dataframe: ' + str(df)
        )
        if return_type == 'dataframe':
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
    ds.get_data(dataset_key="train")
    exit(0)
