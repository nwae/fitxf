import logging
import datasets.arrow_dataset
import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sentence_transformers import InputExample
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from fitxf.utils import Env, Logging


# https://medium.com/@whyamit101/how-to-fine-tune-embedding-models-for-rag-retrieval-augmented-generation-7c5bf08b3c54
class FtEmbeddingDemo:

    def __init__(
            self,
            model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
            dataset_name: str = 'imdb',
            max_sentence_tokens: int = 128,
            logger: Logging | None = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_sentence_tokens = max_sentence_tokens

        self.is_cuda_avail = torch.cuda.is_available()
        self.logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        try:
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

        self.setup()
        return

    def setup(
            self,
    ):
        self.tok_model_path = er.MODELS_PRETRAINED_DIR + '/' + self.model_name
        self.emb_model_path = er.MODELS_PRETRAINED_DIR + '/' + self.model_name

        # Validate model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = self.tok_model_path,
        )
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path = self.emb_model_path,
        )
        self.logger.info(
            'Tokenizer and model loaded successfully for base model "' + str(self.model_name) + '"'
        )

        # Load your dataset (replace with your specific dataset)
        self.logger.info('Loading dataset "' + str(self.dataset_name) + '"...')
        self.dataset = load_dataset(self.dataset_name)
        self.logger.info(
            'Dataset keys: ' + str(self.dataset.keys()) + ', train length ' + str(len(self.dataset["train"]))
            + ', test length ' + str(len(self.dataset["test"])) + ', type "' + str(type(self.dataset["train"])) + '"'
        )
        self.logger.info(self.dataset["train"][0])  # Inspect the data structure

        self.logger.info('Preprocessing dataset "' + str(self.dataset_name) + '"...')
        self.tokenized_data = self.dataset.map(self.preprocess, batched=True)
        self.logger.info(self.tokenized_data["train"][0])  # Verify the tokenized output

        self.logger.info('Creating +/- pairs...')
        data_pairs = self.create_pairs(data=self.tokenized_data["train"])
        self.logger.info(f"Number of pairs: {len(data_pairs)}")
        return

    def preprocess(
            self,
            x,
    ):
        return self.tokenizer(
            x["text"],
            truncation = True,
            padding = "max_length",
            max_length = self.max_sentence_tokens,
        )

    # Define a function to create pairs
    def create_train_examples(
            self,
            data: datasets.arrow_dataset.Dataset = None,
    ) -> list:
        # positive_pairs = [(x['text'], x['similar_text']) for x in data]
        # negative_pairs = [(x['text'], x['unrelated_text']) for x in data]
        # return positive_pairs + negative_pairs

        # Define positive and negative pairs
        train_examples = [
            InputExample(texts=["Apple", "Orange"], label=0.9),
            InputExample(texts=["Apple", "Sun"], label=0.1)
        ]
        print(f"Prepared {len(train_examples)} training examples.")
        return train_examples

    def fine_tune(
            self,
            epochs: int = 100,
            warmup_steps: int = 100,
    ):
        train_examples = self.create_train_examples()
        # Create a DataLoader for the training examples
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

        # Define the loss function
        train_loss = losses.CosineSimilarityLoss(self.model)

        # Fine-tune the model
        self.model.fit(
            train_objectives = [(train_dataloader, train_loss)],
            epochs = epochs,
            warmup_steps = warmup_steps,
        )
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    er = Env()
    Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env.fitxf.math.ut')
    demo = FtEmbeddingDemo(
        logger = lgr,
    )

    exit(0)
