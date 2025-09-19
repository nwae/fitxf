import torch
import torch.nn as nn
from torch.nn import Transformer
from fitxf.math.fit.arc.blocks.PositionalEncoding import PositionalEncoding
from fitxf.math.fit.arc.blocks.TokenEmbedding import TokenEmbedding
import logging
import math
import numpy as np
from fitxf.utils import Logging


# Seq2Seq Network
# https://towardsdatascience.com/transformers-141e32e69591
class Seq2SeqTransformer(nn.Module):

    def __init__(
            self,
            num_encoder_layers,
            num_decoder_layers,
            emb_size,
            nhead,
            src_vocab_size,
            tgt_vocab_size,
            emb_correction = None,
            dim_feedforward = 512,
            dropout = 0.1,
            logger = None,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Причиной множения с math.sqrt(self.emb_size) является такого
        #    "The reason we increase the embedding values before the addition is to
        #     make the positional encoding relatively smaller. This means the original
        #     meaning in the embedding vector won’t be lost when we add them together."
        # с этого сайта https://stackoverflow.com/questions/56930821/why-does-embedding-vector-multiplied-by-a-constant-in-transformer-model
        self.emb_size = emb_size
        self.emb_cor = emb_correction if emb_correction is not None else math.sqrt(self.emb_size)
        self.logger = logger if logger is not None else logging.getLogger()

        self.transformer = Transformer(
            d_model            = self.emb_size,
            nhead              = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward    = dim_feedforward,
            dropout            = dropout,
        )
        # Linear layer as "inverse map" of embedding back to token ID
        self.generator = nn.Linear(in_features=self.emb_size, out_features=tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(
            vocab_size = src_vocab_size,
            emb_size   = self.emb_size,
            emb_cor    = self.emb_cor,
        )
        self.tgt_tok_emb = TokenEmbedding(
            vocab_size = tgt_vocab_size,
            emb_size   = self.emb_size,
            emb_cor    = self.emb_cor,
        )
        self.positional_encoding = PositionalEncoding(
            emb_size,
            dropout = dropout,
            logger = self.logger,
        )
        return

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: torch.Tensor,
            tgt_mask: torch.Tensor,
            src_padding_mask: torch.Tensor,
            tgt_padding_mask: torch.Tensor,
            memory_key_padding_mask: torch.Tensor
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src         = src_emb,
            tgt         = tgt_emb,
            src_mask    = src_mask,
            tgt_mask    = tgt_mask,
            memory_mask = None,
            src_key_padding_mask    = src_padding_mask,
            tgt_key_padding_mask    = tgt_padding_mask,
            memory_key_padding_mask = memory_key_padding_mask
        )
        self.logger.debug('Dimension of transformer outs: ' + str(outs.size()))
        self.logger.debug(outs)
        return self.generator(outs)

    def encode(
            self,
            src: torch.Tensor,
            src_mask: torch.Tensor
    ):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(
            self,
            tgt: torch.Tensor,
            # this is the "context"/"state" returned from the encoder of the Transformer
            memory: torch.Tensor,
            tgt_mask: torch.Tensor
    ):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            pad_idx: int,
    ):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(sz=tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.DEVICE).type(torch.bool)

        src_padding_mask = (src == pad_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class Demo:

    def __init__(self):
        self.vcb_size = 7
        self.n_words = 5
        self.seq2seqTf = Seq2SeqTransformer(
            num_encoder_layers = 1,
            num_decoder_layers = 1,
            emb_size = 2,
            nhead = 2,
            src_vocab_size = self.vcb_size,
            tgt_vocab_size = self.vcb_size
        )
        return

    def demo(self):
        src = torch.randint(low=1, high=self.vcb_size, size=(3,self.n_words))
        tgt = torch.randint(low=1, high=self.vcb_size, size=(3,self.n_words))
        for X in (src, tgt,):
            for x in X:
                l0 = np.random.randint(low=1, high=3+1)
                x[-l0:] *= 0
        print('src:\n', src)
        print('tgt\n', tgt)
        src_msk, tgt_msk, src_padding_msk, tgt_padding_msk = self.seq2seqTf.create_mask(src=src, tgt=tgt, pad_idx=0)
        print('src mask:\n', src_msk)
        print('tgt mask:\n', tgt_msk)

        y = self.seq2seqTf.forward(
            src=src, tgt=tgt, src_mask=src_msk, tgt_mask=tgt_msk,
            src_padding_mask=src_padding_msk, tgt_padding_mask=tgt_padding_msk, memory_key_padding_mask=src_padding_msk
        )
        print('Forward:\n', y)
        print('Forward size', y.size())

        y_enc = self.seq2seqTf.encode(src=src, src_mask=src_msk)
        print('Encode:\n', y_enc)
        y_dec = self.seq2seqTf.decode(tgt=tgt, memory=y_enc, tgt_mask=tgt_msk)
        print('Decode:\n', y_dec)
        return


if __name__ == '__main__':
    logger = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)

    res = Demo().demo()
    exit(0)
