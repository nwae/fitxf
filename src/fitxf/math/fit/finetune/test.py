import logging
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from fitxf.utils import Env, Logging


# https://medium.com/@whyamit101/how-to-fine-tune-embedding-models-for-rag-retrieval-augmented-generation-7c5bf08b3c54
class FtEmbeddingDemo:

    def __init__(
            self,
            model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
            dataset_name: str = 'imdb',
            logger: Logging | None = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.model_name = model_name
        self.dataset_name = dataset_name

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
        self.logger.info(self.dataset["train"][0])  # Inspect the data structure

        self.logger.info('Preprocessing dataset "' + str(self.dataset_name) + '"...')
        self.tokenized_data = self.dataset.map(self.preprocess, batched=True)
        self.logger.info(self.tokenized_data["train"][0])  # Verify the tokenized output
        return

    def preprocess(
            self,
            example,
    ):
        return self.tokenizer(
            example["text"],
            truncation = True,
            padding = "max_length",
            max_length = 128,
        )


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    er = Env()
    Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env.fitxf.math.ut')
    demo = FtEmbeddingDemo(
        logger = lgr,
    )

    exit(0)
