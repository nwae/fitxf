import logging
import torch
import os
import re
import traceback
import requests
import numpy as np
from io import BytesIO
from fitxf import TensorUtils
from PIL import Image
from fitxf.math.lang.encode.LangModelInterface import LangModelInterface as LmInterface
from transformers import AutoImageProcessor, AutoModel
from fitxf.math.utils.Env import Env
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Lock import Lock


class ImgPt(LmInterface):

    RETURN_TENSORS = 'np'

    DEFAULT_MODEL_NAME = 'google/vit-base-patch16-224'
    DEFAULT_CHANNEL = 'RGB'

    def __init__(
            self,
            model_name = None,
            cache_folder = None,
            include_tokenizer = False,
            logger = None,
    ):
        super().__init__(
            model_name = model_name,
            cache_folder = cache_folder,
            include_tokenizer = include_tokenizer,
            logger = logger,
        )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tensor_utils = TensorUtils(logger=self.logger)
        self.use_serverless = False

        self.__mutex_name_model = 'modelpt'
        self.__lock_mutexes = Lock(mutex_names=[self.__mutex_name_model], logger=self.logger)

        if self.model_name is None:
            self.model_name = self.DEFAULT_MODEL_NAME

        # User may pass in model downloaded path
        if os.path.isdir(str(self.model_name)):
            self.model_path = self.model_name
        else:
            self.model_path = self.cache_folder + '/' + self.model_name

        assert os.path.isdir(self.model_path), 'Not a directory "' + str(self.model_path) + '"'
        self.logger.info('Model name "' + str(self.model_name) + '" path "' + str(self.model_path) + '"')

        self.logger.info(
            'Lang model "' + str(self.model_name) + '" with cache folder "' + str(self.cache_folder)
            + '", name_or_path "' + str(self.model_path) + '", device "' + str(self.device) + '"'
        )
        self.processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path = self.model_path,
        )
        self.logger.info(
            'OK processor for model "' + str(self.model_path) + '", cache folder "' + str(self.cache_folder) + '"'
        )
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path = self.model_path,
        ).to(self.device)
        self.logger.info(
            'OK Model "' + str(self.model_path) + '", cache folder "' + str(self.cache_folder) + '"'
        )
        return

    def download_image(
            self,
            url,
    ):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert(self.DEFAULT_CHANNEL)
            # img.save('img_' + str(i) + '.bmp')
            np_image = np.array(img)
            # TODO resize
            return np_image
        except Exception as ex:
            errmsg = 'Failed to get image from URL "' + str(url) + '": ' + str(ex) \
                     + ' Stack trace: ' + str(traceback.format_exc())
            self.logger.error(errmsg)
            raise Exception(errmsg)

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
        embeddings = []
        for img_xxx in content_list:
            if type(img_xxx) is str:
                if re.match(pattern="^http", string=img_xxx, flags=re.IGNORECASE):
                    img_file_type = 'url'
                else:
                    img_file_type = 'file'

                if img_file_type == 'url':
                    img_data = self.download_image(url=img_xxx)
                else:
                    img = Image.open(img_xxx)
                    img_data = img

                # TODO Need to reshape?
                # img.thumbnail(size=self.default_shape)
                # img_data = img.resize(size=self.default_shape)
                img_data = np.array(img_data)

                self.logger.info(
                    'Converted non array image data "' + str(img_xxx) + '" type "' + str(img_file_type)
                    + '" to type "' + str(type(img_data)) + '"' # numpy array shape ' + str(img_data.shape)
                )
            elif type(img_data) in [torch.Tensor, np.ndarray]:
                img_data = img_xxx
            else:
                raise Exception('Unsupported image type "' + str(type(img_data)) + '"')
            emb = self.encode_image(
                image = img_data,
                return_tensors = return_tensors,
            )
            self.logger.info('Encoded image to shape ' + str(emb.shape))
            embeddings.append(emb)
        if return_tensors == 'pt':
            return (torch.vstack(embeddings), None) if return_attnmasks else torch.vstack(embeddings)
        else:
            return (np.vstack(embeddings), None) if return_attnmasks else np.vstack(embeddings)

    def encode_image(
            self,
            image: np.ndarray,
            # permitted values 'last_hidden_state_mean', 'last_hidden_state_cls',  'pooler_output'
            embedding_method: str = 'last_hidden_state_mean',
            return_tensors = 'pt',
    ):
        assert embedding_method in ['last_hidden_state_mean', 'last_hidden_state_cls',  'pooler_output']
        self.logger.info(
            'Type of image to encode is "' + str(type(image)) + '", return tensors "' + str(return_tensors)
            # + '", shape ' + str(image.shape)
        )
        inputs = self.processor(
            image,
            return_tensors = 'pt',
        ).to(self.device)
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        if embedding_method in ['last_hidden_state_mean']:
            torch_embedding = last_hidden_state.mean(dim=1)
        elif embedding_method in ['last_hidden_state_cls']:
            torch_embedding = last_hidden_state[:, 0, :]
        else:
            self.logger.warning(
                'This method produces varying values for the same inputs!! This is due to uninitialized pooler weights.'
            )
            # An alternative representation is indeed the pooler_output,
            # which takes the embedding of the first special CLS token from the last_hidden_state,
            # and applies a layernorm to it as seen here
            # https://github.com/huggingface/transformers/blob/95754b47a6d4fbdad3440a45762531e8c471c528/src/transformers/models/clip/modeling_clip.py#L865
            torch_embedding = outputs.pooler_output

        if return_tensors == 'pt':
            return torch_embedding
        else:
            return torch_embedding.detach().numpy()


if __name__ == '__main__':
    er = Env()
    Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env.fitxf.math.ut')
    urls = [
        'http://images.cocodataset.org/val2017/000000039769.jpg',
        'https://img.freepik.com/free-photo/large-set-isolated-vegetables-white-background_485709-44.jpg?t=st=1729854477~exp=1729858077~hmac=55e5cf830d6a663aac9875a3de6ee54bf4a38ab8cec3400de6732bd94ea2d444&w=740',
        'https://upload.wikimedia.org/wikipedia/commons/b/b3/Blackpink_Ros%C3%A9_Rimowa_1.jpg',
    ]
    # image = Image.open(requests.get(urls[0], stream=True).raw)
    # np_image = np.array(image)
    # print(image, np_image)
    # print('Image type "' + str(type(image)) + '", shape ' + str(np_image.shape))

    pt = ImgPt(
        cache_folder = er.MODELS_PRETRAINED_DIR,
        logger = Logging.get_default_logger(log_level=logging.INFO, propagate=False),
    )
    embed_repeats = []

    # check that embedding remains the same value for the same inputs
    for i in range(2):
        embed = pt.encode(
            text_list = urls,
            return_tensors = 'np',
            return_attnmasks = False,
        )
        embed_repeats.append(embed)
        print(i, embed[:,-5:])
        print(i, type(embed), embed.shape)
        if i > 0:
            assert np.sum(embed**2 - embed_repeats[-1]**2) < 0.0000000001
    exit(0)
