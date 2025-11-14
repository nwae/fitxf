from torch import Tensor
import torch
import torch.nn as nn
import logging
from fitxf.math.utils.Logging import Logging


class TokenEmbedding(nn.Module):

    def __init__(
            self,
            vocab_size,
            emb_size,
            # Embedding correction
            emb_cor = None,
            logger = None,
    ):
        super(TokenEmbedding, self).__init__()
        self.logger = logger if logger else logging.getLogger()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.logger.debug('Embedding start weights:')
        self.logger.debug(self.embedding.weight)
        self.emb_size = emb_size
        self.emb_cor = emb_cor if emb_cor is not None else 1.0

    def get_trained_token_embeddings(self):
        return torch.Tensor(self.embedding.weight)

    def forward(
            self,
            tokens: Tensor,
    ):
        # убедиться что токены являются цедыми числами через long()
        # Причиной множения с math.sqrt(self.emb_size) является такого
        #    "The reason we increase the embedding values before the addition is to
        #     make the positional encoding relatively smaller. This means the original
        #     meaning in the embedding vector won’t be lost when we add them together."
        # с этого сайта https://stackoverflow.com/questions/56930821/why-does-embedding-vector-multiplied-by-a-constant-in-transformer-model
        # self.logger.debug('Passing in tokens of size ' + str(tokens.size()) + ':\n' + str(tokens.long()))
        embed = self.embedding(tokens.long()) * self.emb_cor
        # self.logger.debug('Embedding of tokens:\n' + str(embed))
        return embed


class TokenEmbeddingUnitTest:

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        tokens = torch.FloatTensor([
            [1, 2, 3, 4],
            [0, 2, 4, 1],
            [2, 4, 1, 3],
        ])
        layer = TokenEmbedding(
            vocab_size = 5,
            emb_size = 2,
            emb_cor = 1.,
            logger = lgr,
        )
        te = layer.get_trained_token_embeddings()
        self.logger.info('Token fixed embeddings: ' + str(te))
        # Default transformer dimension is (sentence length, batch, ..), so we need
        # swap the first 2 dimensions
        emb = layer.forward(tokens=tokens)
        # check back using network weights
        emb2 = torch.Tensor([torch.vstack([te[int(tok)] for tok in tokens_i]).tolist() for tokens_i in tokens])
        # emb2 = torch.reshape(emb2, emb.shape)
        self.logger.info('Forward: ' + str(emb) + ', shape ' + str(emb.shape))
        self.logger.info('Manual: ' + str(emb2) + ', shape ' + str(emb2.shape))

        is_passed = torch.sum((emb2 - emb) ** 2).item() < 0.000000001
        self.logger.info('OK = ' + str(is_passed))
        assert is_passed
        self.logger.info('TESTS PASSED')
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    TokenEmbeddingUnitTest(logger=lgr).test()
    exit(0)
