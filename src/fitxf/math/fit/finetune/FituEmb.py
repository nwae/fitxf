import logging
import torch
import re
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from fitxf.math.data.Dataset import DatasetUtil
from fitxf.utils import Logging, Env


class FituEmb:

    def __init__(
            self,
            model_name_or_path: str,
            device_if_no_cuda: str = 'mps',
            logger: Logging | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device_if_no_cuda = device_if_no_cuda
        self.logger = logger if logger is not None else logging.getLogger()

        self.device = 'cuda' if torch.cuda.is_available() else self.device_if_no_cuda
        self.logger.info('Using device "' + str(self.device) + '"')

        # 1. Load a model to finetune with 2. (Optional) model card data
        self.model = SentenceTransformer(
            device = self.device,
            model_name_or_path = model_name_or_path,
            # model_card_data = SentenceTransformerModelCardData(
            #     language = "en",
            #     license = "apache-2.0",
            #     model_name = "MPNet base trained on AllNLI triplets",
            # )
        )
        # [lgr.info('Model parameter #' + str(i) + ': ' + str(p)) for i, p in enumerate(model.parameters())]
        return

    def fine_tune(
            self,
            # e.g. [('sentence-transformers/all-nli', 'triplet'), ('sentence-transformers/natural-questions', '')]
            dataset_paths_names: list[tuple],
            Loss_funcs: list,
            train_dataset_select_range: int = 0,
            epochs: int = 100,
            batch_size: int = 16,
            learn_rate: float = 2e-5,
            # gradually ramp up to learn_rate during this first % of epoch
            warmup_ratio: float = 0.1,
            # GPU can run on FP16
            gpu_fp16: bool = False,
            # GPU supports BF16
            gpu_bf16: bool = False,
            output_dir: str | None = None,
    ):
        assert len(dataset_paths_names) == len(Loss_funcs)

        train_ds, eval_ds, test_ds, loss_funcs = {}, {}, {}, {}
        for i, t in enumerate(dataset_paths_names):
            dpath, dname = t
            dkey = dpath + '-' + dname if dname != '' else dpath

            du = DatasetUtil(logger=self.logger)
            self.logger.info('#' + str(i) + ' ' + str(t) + ' Start downloading dataset...')
            # 3. Load a dataset to finetune on
            du.download(dataset_path=dpath, dataset_name=dname)

            loss_funcs[dkey] = Loss_funcs[i]
            train_ds[dkey] = du.get_data(dataset_key="train", select_range=train_dataset_select_range)
            # Eval & test datasets may not exist
            if du.is_dataset_exist(dataset_key='dev'):
                eval_ds[dkey] = du.get_data(dataset_key="dev")
            if du.is_dataset_exist(dataset_key='test'):
                test_ds[dkey] = du.get_data(dataset_key="test")

            self.logger.info('#' + str(i) + ' ' + str(t) + ' using loss function ' + str(loss_funcs[dkey]))
            self.logger.info('#' + str(i) + ' ' + str(t) + ' train dataset 0-10: ' + str(train_ds[dkey][0:10]))
            self.logger.info('#' + str(i) + ' ' + str(t) + ' train dataset type "' + str(type(train_ds[dkey])) + '"')
            self.logger.info('#' + str(i) + ' ' + str(t) + ' train dataset length "' + str(len(train_ds[dkey])) + '"')

        run_name = re.sub(pattern=".*/", repl="", string=self.model_name_or_path) \
                   + "---" + '___'.join(['-'.join(t) for t in dataset_paths_names])
        self.logger.info('Using run name "' + str(run_name) + '"')

        # Optional training arguments
        train_args = SentenceTransformerTrainingArguments(
            output_dir = output_dir,
            num_train_epochs = epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            learning_rate = learn_rate,
            warmup_ratio = warmup_ratio,
            fp16 = gpu_fp16,
            bf16 = gpu_bf16,
            batch_sampler = BatchSamplers.NO_DUPLICATES,
            # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            # Optional tracking/debugging parameters:
            eval_strategy = "steps",
            eval_steps = 100,
            save_strategy = "steps",
            save_steps = 100,
            save_total_limit = 2,
            logging_steps = 100,
            # Will be used in W&B if `wandb` is installed
            run_name = run_name,
        )

        for i, t in enumerate(dataset_paths_names):
            dpath, dname = t
            dkey = dpath + '-' + dname if dname != '' else dpath
            self.logger.info('#' + str(i) + ' ' + str(t) + ' start triplet evaluation...')
            if dkey in eval_ds.keys():
                dev_evaluator = TripletEvaluator(
                    anchors = eval_ds[dkey]["anchor"],
                    positives = eval_ds[dkey]["positive"],
                    negatives = eval_ds[dkey]["negative"],
                    name = dkey + "-dev",
                )
                dev_evaluator(self.model)
            else:
                self.logger.warning('Evaluation dataset for dataset key "' + str(dkey) + '" is not available')

        trainer = SentenceTransformerTrainer(
            model = self.model,
            args = train_args,
            train_dataset = train_ds,
            eval_dataset = eval_ds,
            loss = loss_funcs,
            # evaluator = dev_evaluator,
        )
        trainer.train()

        for i, t in enumerate(dataset_paths_names):
            dpath, dname = t
            dkey = dpath + '-' + dname if dname != '' else dpath
            self.logger.info('#' + str(i) + ' ' + str(t) + ' start triplet evaluation...')
            if dkey in test_ds.keys():
                test_evaluator = TripletEvaluator(
                    anchors = test_ds[dkey]["anchor"],
                    positives = test_ds[dkey]["positive"],
                    negatives = test_ds[dkey]["negative"],
                    name = dkey + "-test",
                )
                test_evaluator(self.model)

        if output_dir is not None:
            self.model.save_pretrained(output_dir + "/final")

        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    er = Env(logger=lgr)
    Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env.fitxf.math.ut')

    fitu = FituEmb(
        model_name_or_path = er.MODELS_PRETRAINED_DIR + '/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        device_if_no_cuda = 'mps',
        logger = lgr,
    )
    mnrl_loss = MultipleNegativesRankingLoss(fitu.model)
    fitu.fine_tune(
        dataset_paths_names = [
            ('sentence-transformers/all-nli', 'triplet'),
            ('sentence-transformers/natural-questions', ''),
        ],
        Loss_funcs = [
            mnrl_loss,
            mnrl_loss,
        ],
        train_dataset_select_range = 20,
        epochs = 1,
        output_dir = 'tmp/finetune',
    )
    exit(0)
