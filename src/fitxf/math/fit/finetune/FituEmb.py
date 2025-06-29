import logging
import torch
import re
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    # SentenceTransformerModelCardData,
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

        self.dataset_utils = DatasetUtil(
            logger = self.logger,
        )

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
            # e.g. 'sentence-transformers/all-nli'
            dataset_path: str,
            # e.g. 'triplet'
            dataset_name: str,
            train_dataset_select_range: int = 0,
            Loss_func: torch.nn.Module = MultipleNegativesRankingLoss,
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
        self.logger.info('Start downloading dataset "' + str(dataset_path) + '"...')
        # 3. Load a dataset to finetune on
        self.dataset_utils.download(dataset_path=dataset_path, dataset_name=dataset_name)

        train_dataset = self.dataset_utils.get_data(dataset_key="train", select_range=train_dataset_select_range)
        eval_dataset = self.dataset_utils.get_data(dataset_key="dev")
        test_dataset = self.dataset_utils.get_data(dataset_key="test")
        self.logger.info('Test dataset: ' + str(test_dataset))
        self.logger.info('Test dataset 0-10: ' + str(test_dataset[0:10]))
        self.logger.info('Test dataset type "' + str(type(test_dataset)) + '"')
        self.logger.info('Test dataset length "' + str(len(test_dataset)) + '"')
        # raise Exception('asdf')

        self.logger.info('Using loss function ' + str(Loss_func))
        loss = Loss_func(self.model)

        run_name = re.sub(pattern=".*/", repl="", string=self.model_name_or_path) \
                   + "-" + re.sub(pattern=".*/", repl="", string=dataset_path) \
                   + "-" + dataset_name
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

        dev_evaluator = TripletEvaluator(
            anchors = eval_dataset["anchor"],
            positives = eval_dataset["positive"],
            negatives = eval_dataset["negative"],
            name = "all-nli-dev",
        )
        dev_evaluator(self.model)

        trainer = SentenceTransformerTrainer(
            model = self.model,
            args = train_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            loss = loss,
            evaluator = dev_evaluator,
        )
        trainer.train()

        test_evaluator = TripletEvaluator(
            anchors = test_dataset["anchor"],
            positives = test_dataset["positive"],
            negatives = test_dataset["negative"],
            name = "all-nli-test",
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
    fitu.fine_tune(
        dataset_path = 'sentence-transformers/all-nli',
        dataset_name = 'triplet',
        train_dataset_select_range = 100,
        epochs = 1,
        output_dir = 'tmp/finetune',
    )
    exit(0)
