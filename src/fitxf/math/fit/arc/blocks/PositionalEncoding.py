from torch import Tensor
import torch
import torch.nn as nn
import math
import logging


# https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):

    def __init__(
            self,
            # must be even number, so that sin & cos series are equal length
            emb_size,
            dropout,
            # fix length objects (e.g. max "tokens" in "sentences")
            maxlen = 5000,
            logger = None,
    ):
        super(PositionalEncoding, self).__init__()
        self.logger = logger if logger is not None else logging.getLogger()

        assert emb_size % 2 == 0, 'Embedding size must be even number, not ' + str(emb_size)

        # weights on the embedding vector, so means same length as embedding size, decrease with index
        # e.g. for emb_size = 128, it decreases from 1.0 to 0.00011548 at position-127
        # tensor([1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01,...
        #       ....
        #       1.7783e-04, 1.5399e-04, 1.3335e-04, 1.1548e-04])
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        self.logger.debug('Den: ' + str(den))
        # position of the "word" (token embedding) in "sentence"
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        self.logger.debug('Pos: ' + str(pos))

        pos_embedding = torch.zeros((maxlen, emb_size))
        # 0::2 means the array of 0 mod(2) numbers or [0, 2, 4, ...]
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        # print(torch.sin(pos * den))
        # 1::3 means the array of 1 mod(2) numbers or [1, 3, 5, ...]
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # print(torch.cos(pos * den))
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.logger.debug('Position embedding: ' + str(pos_embedding))

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        return

    def forward(
            self,
            token_embedding: Tensor,
    ):
        self.logger.debug(
            'Token embedding shape ' + str(token_embedding.shape)
            + ', pos embedding shape ' + str(self.pos_embedding.shape)
        )
        self.logger.debug(
            'Add positional embedding: ' + str(self.pos_embedding[:token_embedding.size(0), :])
            + ' to token embedding: ' + str(token_embedding)
        )
        # Default transformer dimension is (sentence length, batch, ..), so we take the first index
        # and match the sentence length with the embedding positions
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    n_batch, sentence_len, embd_sz = 2, 3, 4
    pe = PositionalEncoding(
        emb_size = embd_sz,
        dropout  = 0.,
        maxlen = sentence_len,
    )
    # 11 samples of fixed length (=20) sentences, of token embedding dim 8
    # y = pe.forward(token_embedding=torch.randint(1000, (11, 20, embd_sz)))
    x = torch.ones(n_batch, sentence_len, embd_sz)
    for i in range(n_batch):
        x[i, :, :] *= (i + 1)
    print(x)
    # Default transformer dimension is (sentence length, batch, ..), so we need
    # swap the first 2 dimensions
    y = pe.forward(token_embedding=x.transpose(0, 1))
    print(y.transpose(0, 1))
    exit(0)
