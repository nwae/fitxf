import torch
import logging
import os
from fitxf.math.lang.encode.LangModelInterface import LangModelInterface
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from fitxf.math.utils.Env import Env
from fitxf.math.utils.Logging import Logging


#
# Leader Board: https://huggingface.co/spaces/mteb/leaderboard
#
class LangModelPt(LangModelInterface):

    DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

    def __init__(
            self,
            model_name: str = None,
            cache_folder: str = None,
            include_tokenizer: bool = False,
            logger = None,
    ):
        super().__init__(
            cache_folder = cache_folder,
            model_name = model_name,
            include_tokenizer = include_tokenizer,
            logger = logger,
        )

        self.model_name = self.DEFAULT_MODEL if self.model_name is None else self.model_name
        self.logger.info(
            'Model name "' + str(self.model_name) + '", cache folder "' + str(self.cache_folder) + '"'
        )

        # User may pass in model downloaded path
        if os.path.isdir(str(self.model_name)):
            self.model_path = self.model_name
        else:
            self.model_path = self.cache_folder + '/' + self.model_name

        assert os.path.isdir(self.model_path), 'Not a directory "' + str(self.model_path) + '"'
        self.logger.info('Model name "' + str(self.model_name) + '" path "' + str(self.model_path) + '"')

        self.logger.info(
            'Lang model "' + str(self.model_name) + '" with cache folder "' + str(self.cache_folder)
            + '", include tokenizer "' + str(self.include_tokenizer)
            + '", name_or_path "' + str(self.model_path) + '"'
        )
        if self.include_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path = self.model_path,
                # cache_folder = self.cache_folder,
            )
            self.logger.info(
                'OK Tokenizer for model "' + str(self.model_path) + '", cache folder "' + str(self.cache_folder) + '"'
            )
            self.model = AutoModel.from_pretrained(
                # hugging face will know to use the cache folder above, without specifying here it seems
                pretrained_model_name_or_path = self.model_path
            )
            self.logger.info(
                'OK Model "' + str(self.model_path) + '", cache folder "' + str(self.cache_folder) + '"'
            )
        else:
            self.model = SentenceTransformer(
                model_name_or_path = self.model_path,
                cache_folder = self.cache_folder,
            )
        return

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(
            self,
            content_list,
            # max length has got no meaning
            maxlen = None,
            return_tensors = 'pt',
            # does not apply here since we can't see the tokenization
            return_attnmasks = False,
            params_other = None,
    ):
        if self.include_tokenizer:
            self.logger.debug('Calculating embedding using tokenizer')
            embeddings_tensor = None
            for txt in content_list:
                # Tokenize sentences
                encoded_input = self.tokenizer(txt, padding=True, truncation=True, return_tensors='pt')

                # Compute token embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                # Perform pooling. In this case, max pooling.
                attn_masks = encoded_input['attention_mask']
                embedding = self.mean_pooling(model_output=model_output, attention_mask=attn_masks)
                if embeddings_tensor is not None:
                    embeddings_tensor = torch.cat((embeddings_tensor, embedding), dim=0)
                else:
                    embeddings_tensor = embedding

            attn_masks = None
            if return_tensors == 'pt':
                return (embeddings_tensor, attn_masks) if return_attnmasks else embeddings_tensor
            else:
                embedding_np = embeddings_tensor.cpu().detach().numpy()
                return (embedding_np, attn_masks) if return_attnmasks else embedding_np
        else:
            self.logger.debug('Calculating embedding using single sentence transformer wrapper')
            embedding = self.model.encode(
                sentences = content_list,
            )
            # TODO
            #    How to get this? Since this depends on the tokenizer used by the language model,
            #    this means we cannot calculate on our own by tokenizing.
            attn_masks = None

            if return_tensors == 'pt':
                pt_array = torch.from_numpy(embedding)
                return (pt_array, attn_masks) if return_attnmasks else pt_array
            else:
                return (embedding, attn_masks) if return_attnmasks else embedding


if __name__ == '__main__':
    er = Env()
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)

    lm = LangModelPt(
        model_name = 'intfloat/multilingual-e5-small',
        cache_folder = er.MODELS_PRETRAINED_DIR,
        logger = lgr,
    )
    text_list = [
        '칠리 페퍼', '와사비', '머스타드',
        '케이크', '도넛', '아이스크림',
    ]
    labels = ['hot', 'hot', 'hot', 'sweet', 'sweet', 'sweet']
    lm.visualize_embedding(
        encoding_np = lm.encode(content_list=text_list, return_tensors='np'),
        labels_list = labels,
    )

    print('rps', lm.speed_test(sentences=text_list, min_rounds=200))
    exit(0)
