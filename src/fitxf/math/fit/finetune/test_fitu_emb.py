import logging
import torch
from fitxf.math.data.Dataset import DatasetUtil
from fitxf.utils import Logging, Env

# See https://sbert.net/docs/sentence_transformer/training_overview.html

lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
er = Env(logger=lgr)
Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env.fitxf.math.ut')
dataset_utils = DatasetUtil(logger=lgr)

model_name = 'intfloat/multilingual-e5-small'
model_path = er.MODELS_PRETRAINED_DIR + '/' + model_name
dataset_path, dataset_name = 'sentence-transformers/all-nli', 'triplet'

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

device = 'cuda' if torch.cuda.is_available() else 'mps'
lgr.info('Using device "' + str(device) + '"')

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    device = device,
    # "microsoft/mpnet-base",
    model_name_or_path = model_path,
    # model_card_data = SentenceTransformerModelCardData(
    #     language = "en",
    #     license = "apache-2.0",
    #     model_name = "MPNet base trained on AllNLI triplets",
    # )
)
# [lgr.info('Model parameter #' + str(i) + ': ' + str(p)) for i, p in enumerate(model.parameters())]
# raise Exception('asdf')

lgr.info('Start downloading dataset "' + str(dataset_path) + '"...')
# 3. Load a dataset to finetune on
dataset_utils.download(dataset_path=dataset_path, dataset_name=dataset_name)

train_dataset = dataset_utils.get_data(dataset_key="train", select_range=1000)
eval_dataset = dataset_utils.get_data(dataset_key="dev")
test_dataset = dataset_utils.get_data(dataset_key="test")
lgr.info('Test dataset: ' + str(test_dataset))
lgr.info('Test dataset 0-10: ' + str(test_dataset[0:10]))
lgr.info('Test dataset type "' + str(type(test_dataset)) + '"')
lgr.info('Test dataset length "' + str(len(test_dataset)) + '"')
# raise Exception('asdf')

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)

output_dir = er.REPO_DIR + "/tmp/finetune"
# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir = output_dir,
    # Optional training parameters:
    num_train_epochs = 1,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    learning_rate = 2e-5,
    warmup_ratio = 0.1,
    fp16 = True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16 = False,  # Set to True if you have a GPU that supports BF16
    batch_sampler = BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy = "steps",
    eval_steps = 100,
    save_strategy = "steps",
    save_steps = 100,
    save_total_limit = 2,
    logging_steps = 100,
    run_name = model_name + "-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors = eval_dataset["anchor"],
    positives = eval_dataset["positive"],
    negatives = eval_dataset["negative"],
    name = "all-nli-dev",
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model = model,
    args = args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    loss = loss,
    evaluator = dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the test set
test_evaluator = TripletEvaluator(
    anchors = test_dataset["anchor"],
    positives = test_dataset["positive"],
    negatives = test_dataset["negative"],
    name = "all-nli-test",
)
test_evaluator(model)

# 8. Save the trained model
model.save_pretrained(output_dir + "/final")

# 9. (Optional) Push it to the Hugging Face Hub
# model.push_to_hub("mpnet-base-all-nli-triplet")

