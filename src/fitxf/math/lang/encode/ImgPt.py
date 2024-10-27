import logging
import torch
import os
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

    def __init__(
            self,
            model_name: str = None,
            cache_folder: str = None,
            include_tokenizer: bool = False,
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
            'Image model "' + str(self.model_name) + '" with cache folder "' + str(self.cache_folder)
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
            img = Image.open(BytesIO(response.content)).convert('RGB')
            # img.save('img_' + str(i) + '.bmp')
            np_image = np.array(img)
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
        for img in content_list:
            if type(img) is str:
                img_data = self.download_image(url=img)
                self.logger.info(
                    'Converted non array image data "' + str(img) + '" to numpy array shape ' + str(img_data.shape)
                )
            else:
                img_data = img
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
            return_tensors = 'pt',
    ):
        self.logger.info(
            'Type of image to encode is "' + str(type(image)) + '", return tensors "' + str(return_tensors)
            + '", shape ' + str(image.shape)
        )
        inputs = self.processor(
            image,
            return_tensors = 'pt',
        ).to(self.device)
        outputs = self.model(**inputs)
        torch_embedding = outputs.pooler_output
        if return_tensors == 'pt':
            return torch_embedding
        else:
            return torch_embedding.detach().numpy()


if __name__ == '__main__':
    er = Env()
    Env.set_env_vars_from_file(env_filepath=er.REPO_DIR + '/.env.fitxf.math.ut')
    urls = [
        'https://img.freepik.com/premium-photo/shakh-plov-cooked-rice-dish-with-raisins-beautiful-plate-islamic-arabic-food_1279579-5074.jpg?w=1800',
        'https://img.freepik.com/premium-psd/tasty-fried-vegetable-rice-plate-isolated-transparent-background_927015-3126.jpg?w=1480',
    ]
    # image = Image.open(requests.get(urls[0], stream=True).raw)
    # np_image = np.array(image)
    # print(image, np_image)
    # print('Image type "' + str(type(image)) + '", shape ' + str(np_image.shape))

    pt = ImgPt(
        cache_folder = er.MODELS_PRETRAINED_DIR,
        logger = Logging.get_default_logger(log_level=logging.INFO, propagate=False),
    )
    embed = pt.encode(
        content_list = urls,
        return_tensors = 'np',
        return_attnmasks = False,
    )
    print(embed)
    print(type(embed), embed.shape)
