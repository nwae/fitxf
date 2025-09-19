import logging
import math
import os
import gc
import torch
import torch.nn as nn
import pandas as pd
# from torch.utils.data import DataLoader
# from poc.lang.plaintext.Vocab import Vocab
# from poc.lang.plaintext.TextTransform import TextTransform
from fitxf.math.lang.tokenizer.TokenizerAuto import TokenizerAuto, TokenizerInterface
from fitxf.math.lang.tokenizer.TokenizerUnitTest import TokenizerUnitTest
from fitxf.math.fit.arc.blocks.Seq2SeqTransformer import Seq2SeqTransformer
from timeit import default_timer as timer
from fitxf.utils import Env, Logging, Profiling, CmdLine


"""
GPU Memory is always limited. We solve it by:
  - smaller batch, but accumulate grad
  - multiple separate train sessions, reloading old params
"""


class TranslateSeq2SeqTrfm:

    SAVE_FTYPE_MODEL      = 'model'
    SAVE_FTYPE_STATE_DICT = 'state_dict'

    TEST_MODE_SEQ_LEN = 5

    def __init__(
            self,
            lang_src,
            lang_tgt,
            input_token_max_len: int,
            output_token_max_len: int,
            tokenizer_model_path: str,
            load_old_state = True,
            text_pairs: list | None = None,
            cache_dir = None,
            embed_size = 512,
            test_mode = False,
            logger = None,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.input_token_max_len = input_token_max_len
        self.output_token_max_len = output_token_max_len
        self.tokenizer_model_path = tokenizer_model_path
        self.load_old_state = load_old_state
        self.text_pairs = text_pairs
        self.cache_dir = cache_dir if cache_dir is not None else Env.get_home_download_dir()

        self.emb_size = embed_size
        self.test_mode = test_mode
        self.logger = logger if logger is not None else logging.getLogger()

        # self.vocab_dir = self.get_save_base_dir()
        # self.text_transforms = {}
        self.profiler = Profiling(logger=self.logger)

        self.tokenizer = self.load_tokenizer()
        # self.__init_nlp()
        self.__init_model()
        self.logger.info(
            'Init successful using cache dir "' + str(self.cache_dir)
            # + '", vocab dir "' + str(self.vocab_dir)
            + '", embedding size ' + str(self.emb_size)
        )
        return

    def load_tokenizer(
            self,
    ) -> TokenizerInterface:
        if not os.path.exists(self.tokenizer_model_path):
            assert self.text_pairs is not None
            # Use training data to train tokenizer
            self.logger.info(
                'Tokenizer path does not exist "' + str(self.tokenizer_model_path)
                + '", will train from scratch using ' + str(len(self.text_pairs)) + ' text pairs'
            )
            texts_train_tok = []
            for row in self.text_pairs:
                texts_train_tok.append(row[0])
                texts_train_tok.append(row[1])
            tok_tmp = TokenizerAuto(
                model_name_or_path = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                logger = self.logger,
            )
            tok_tmp.train(
                text_corpus = texts_train_tok,
                save_path = self.tokenizer_model_path,
            )
        else:
            self.logger.info('Tokenizer model path exists "' + str(self.tokenizer_model_path) + '"')

        tokenizer = TokenizerAuto(
            model_name_or_path = self.tokenizer_model_path,
            logger = self.logger,
        )
        self.logger.info(
            'Tokenizer loaded successfully from model path "' + str(self.tokenizer_model_path) + '"'
        )

        # Do some random tests
        TokenizerUnitTest(
            tokenizer = tokenizer,
            logger = self.logger,
        ).test(
            test_langs = [None],
            test_texts = [p[0] for i, p in enumerate(self.text_pairs) if i < 3],
        )
        return tokenizer

    def make_fake_test_data(self, batch_size):
        src_rand = torch.randint(
            low = 0,
            high = self.tokenizer.get_vocab_size(),
            size = (2 * batch_size, self.TEST_MODE_SEQ_LEN),
        )
        self.logger.debug('src rand: ' + str(src_rand))

        # Fake "translation" function
        tgt_rand = src_rand + 10
        tgt_rand = tgt_rand % self.tokenizer.get_vocab_size()

        for a, b in list(zip(src_rand, tgt_rand)):
            self.logger.debug(str(a) + ' --> ' + str(b))

        # 1 batch each for train & eval. transpose to reverse batch into the 2nd index
        # (which is default for pytorch Transformer module)
        ds_trn = [(src_rand[:batch_size].transpose(0, 1), tgt_rand[:batch_size].transpose(0, 1))]
        ds_val = [(src_rand[batch_size:].transpose(0, 1), tgt_rand[batch_size:].transpose(0, 1))]

        self.logger.info(
            'Test dataset train/val shapes ' + str([ds_trn[0][0].shape, ds_trn[0][1].shape])
            + ' and ' + str([ds_val[0][0].shape, ds_val[0][1].shape])
        )
        return ds_trn, ds_val

    # def load_all(
    #         self,
    # ):
    #     if self.test_mode:
    #         pass
    #     else:
    #         # For fine-tuning, prediction, we MUST load from old state
    #         if self.load_old_state:
    #             # Try to load from file
    #             for lng in (self.lang_src, self.lang_tgt):
    #                 txt_trf = TextTransform(
    #                     lang = lng,
    #                     cache_dir = self.cache_dir,
    #                     vocab_dir = self.vocab_dir,
    #                     logger = self.logger,
    #                 )
    #                 txt_trf.load_vocab()
    #                 self.logger.info(
    #                     'Loaded text transform from "' + str(txt_trf.vocab.get_vocab_save_path())
    #                     + '" for lang "' + str(lng) + '" of type "' + str(type(txt_trf)) + '"'
    #                 )
    #                 self.text_transforms[lng] = txt_trf
    #         else:
    #             for lng in (self.lang_src, self.lang_tgt):
    #                 txt_trf = TextTransform(
    #                     lang = lng,
    #                     cache_dir = self.cache_dir,
    #                     vocab_dir = self.vocab_dir,
    #                     logger = self.logger,
    #                 )
    #                 txt_trf.load_vocab()
    #
    #     return

    # def __init_nlp(
    #         self,
    # ):
    #     if self.test_mode:
    #         self.src_vocab_size = 100
    #         self.tgt_vocab_size = 100
    #     else:
    #         self.src_vocab_size = self.text_transforms[self.lang_src].vocab.get_vocab_size()
    #         self.tgt_vocab_size = self.text_transforms[self.lang_tgt].vocab.get_vocab_size()
    #     self.logger.info('Vocab sizes src/tgt ' + str(self.src_vocab_size) + '/' + str(self.tgt_vocab_size))
    #     return

    def __init_model(
            self,
    ):
        torch.manual_seed(0)

        self.nhead = 8
        self.ffn_hid_dim = 512
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3

        self.nn_transformer = Seq2SeqTransformer(
            num_encoder_layers = self.num_encoder_layers,
            num_decoder_layers = self.num_decoder_layers,
            emb_size = self.emb_size,
            # Причиной множения с math.sqrt(self.emb_size) является такого
            #    "The reason we increase the embedding values before the addition is to
            #     make the positional encoding relatively smaller. This means the original
            #     meaning in the embedding vector won’t be lost when we add them together."
            # с этого сайта https://stackoverflow.com/questions/56930821/why-does-embedding-vector-multiplied-by-a-constant-in-transformer-model
            emb_correction = math.sqrt(self.emb_size),
            nhead = self.nhead,
            src_vocab_size = self.tokenizer.get_vocab_size(),
            tgt_vocab_size = self.tokenizer.get_vocab_size(),
            dim_feedforward = self.ffn_hid_dim,
        )

        self.nn_transformer = self.nn_transformer.to(self.device)

        if self.load_old_state:
            # Loading old state always uses state dict
            self.logger.info('Loading model from a previous state..')
            self.__load_state(load_as = self.SAVE_FTYPE_STATE_DICT)
        else:
            self.logger.info('Randomly initializing model parameter weights..')
            # Random initialization of model parameters if not loading from a previous state
            for p in self.nn_transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            self.__init_optimizer()

        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index = self.tokenizer.get_pad_token_id(),
        )
        return

    def __init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.nn_transformer.parameters(),
            lr    = 0.0001,
            betas = (0.9, 0.98),
            eps   = 1e-9
        )

    def get_model_id_name(
            self,
    ):
        model_id_name = 'tl.seq2seq.tfm.emb' + str(self.emb_size) + '.' + str(self.lang_src) + '-' + str(self.lang_tgt)
        return model_id_name

    def get_save_base_dir(
            self,
    ):
        id_name = self.get_model_id_name()
        basedir = self.cache_dir + '/' + str(id_name)
        if not os.path.exists(basedir):
            self.logger.info('Directory does not exist, creating new directory "' + str(basedir) + '"')
            os.mkdir(path=basedir)
        else:
            self.logger.info('Directory already exists "' + str(basedir) + '"')
        return basedir

    def get_save_filepath(
            self,
    ):
        id_name = self.get_model_id_name()
        basedir = self.get_save_base_dir()
        return basedir + '/' + str(id_name) + '.model.state_dict'

    def get_dataloader(
            self,
            batch_size,
            max_samples,
            val_prop = 0.2,
    ):
        assert batch_size > 0, 'Invalid batch size ' + str(batch_size)
        assert val_prop ** 2 < 1, 'Invalid validation proportion ' + str(val_prop)

        if self.test_mode:
            return self.make_fake_test_data(batch_size=batch_size)
        else:
            # just an array of batch tuples, not necessarily the stupid overcomplicated torch DataLoader
            #    e.g. [(X_batch_1, y_batch_1), (X_batch_2, y_batch_2), ...]
            # or if attention masks included
            #    e.g. [(X_batch_1, attn_mask_batch_1, y_batch_1), (X_batch_2, attn_mask_batch_2, y_batch_2), ...]
            # The default batch dimension is (pos, batch) which is why we need to transpose the tensor later
            dl_trn, dl_val = [], []
            i = 0
            val_size = max(1, int(val_prop * batch_size))

            f_ids = self.tokenizer.encode
            l_in, l_out = self.input_token_max_len, self.output_token_max_len
            while i < len(self.text_pairs) - (batch_size + val_size):
                data_round_trn = self.text_pairs[i:(i+batch_size)]
                i += batch_size
                data_round_val = self.text_pairs[i:(i+val_size)]
                i += val_size
                batch_trn_tmp = (
                    torch.LongTensor([f_ids(text=r[0], return_len=l_in) for r in data_round_trn]).transpose(0, -1),
                    torch.LongTensor([f_ids(text=r[1], return_len=l_out) for r in data_round_trn]).transpose(0, -1)
                )
                batch_val_tmp = (
                    torch.LongTensor([f_ids(text=r[0], return_len=l_in) for r in data_round_val]).transpose(0, -1),
                    torch.LongTensor([f_ids(text=r[1], return_len=l_out) for r in data_round_val]).transpose(0, -1)
                )
                # batch_trn_tmp = [[f_ids(text=r[0]) for r in data_round_trn], [f_ids(text=r[1]) for r in data_round_trn]]
                # batch_val_tmp = [[f_ids(text=r[0]) for r in data_round_val], [f_ids(text=r[1]) for r in data_round_val]]
                # self.logger.debug('Round batch train: ' + str(batch_trn_tmp))
                # self.logger.debug('Round batch validation: ' + str(batch_val_tmp))
                # raise Exception('asdf')
                dl_trn.append(batch_trn_tmp)
                dl_val.append(batch_val_tmp)
            # eval_iter = self.corpora_txtpair.get_data_iter()
            # len_corpora = len(eval_iter)
            # if len_corpora > max_samples:
            #     self.logger.warning(
            #         'Corpora length ' + str(len_corpora) + ' exceed max samples ' + str(max_samples)
            #         + ', randomly sample to reduce...'
            #     )
            #     eval_iter = tuple(MathUtils().sample_random_no_repeat(list=eval_iter, n=max_samples))
            #     self.logger.info('New corpora samples iter length ' + str(len(eval_iter)))
            #
            # idx_cut = int(len(eval_iter) * (1-val_prop))
            # eval_iter_trn = eval_iter[0:idx_cut]
            # eval_iter_val = eval_iter[idx_cut:]
            #
            # dl_trn = DataLoader(
            #     eval_iter_trn,
            #     batch_size = batch_size,
            #     # function to convert raw text pair into tokenized/numeralized tensor pairs
            #     collate_fn = self.corpora_txtpair.collate_fn
            # )
            # dl_val = DataLoader(
            #     eval_iter_val,
            #     batch_size = batch_size,
            #     # function to convert raw text pair into tokenized/numeralized tensor pairs
            #     collate_fn = self.corpora_txtpair.collate_fn
            # )
            return dl_trn, dl_val

    def evaluate(
            self,
            mode,
            model,
            # can be any list of tuples (src, tgt), not necessarily the stupid overcomplicated
            # pytorch DataLoader. A simple structure also can work as below:
            #    [(src1_tensor, tgt1_tensor), (src2_tensor, tgt2_tensor), ...]
            dataset_dataloader,
            # logging purposes only
            epoch_no,
            # if in train mode, this should not be None
            optimizer = None,
            accum_steps = 1,
    ):
        assert mode in ['train', 'validate'], 'Mode not permitted "' + str(mode) + '"'
        # Set to train or validate mode, whether to keep gradients and turn on layers like DropOut, etc.
        model.train() if mode == 'train' else model.eval()

        losses = 0
        self.logger.debug(
            'Length of "' + str(mode) + '" dataset dataloader = ' + str(len(dataset_dataloader))
            # + ', dataset ' + str(dataset_dataloader)
        )

        iter = enumerate(dataset_dataloader) if type(dataset_dataloader) in [list, tuple] \
            else enumerate(iterable=dataset_dataloader, start=0)
        for step, batch in iter:
            self.logger.debug('Step ' + str(step) + ', batch ' + str(batch))
            start_accum = (step % accum_steps == 0)
            done_accum = ((step + 1) % accum_steps == 0) or (step + 1 == len(dataset_dataloader))

            src, tgt = batch
            # Move torch Tensor type to either CPU or GPU device
            src = src.to(self.device)
            # Teacher forcing the actual ground truth to decoder
            tgt = tgt.to(self.device)

            self.logger.debug(
                'Step #' + str(step) + ', src shape ' + str(src.shape) + ': ' + str(src)
                + ', target shape ' + str(tgt.shape) + ': ' + str(tgt)
            )

            # take until the second last token ID for every sentence
            tgt_input = tgt[:-1, :]
            # self.logger.info('Target ' + str(tgt))
            # self.logger.info('Target input ' + str(tgt_input))
            # this part a bit messy since we need to call back to data creator
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)
            # self.logger.info('src mask ' + str(src_mask))
            # self.logger.info('tgt mask ' + str(tgt_mask))

            if mode == 'train':
                if start_accum:
                    # reset model paramater gradients before forward pass
                    optimizer.zero_grad()

            self.logger.debug(
                'Starting forward pass epoch no ' + str(epoch_no) + ', step #' + str(step)
                + ' with teacher forcing through model "' + str(type(model))
                + '", from source shape ' + str(src.size())
                + ', target shape ' + str(tgt_input.size()) + ', source mask shape ' + str(src_mask.size())
                + ', target mask shape ' + str(tgt_mask.size()) + ', source padding mask shape '
                + str(src_padding_mask.size()) + ', target padding mask shape ' + str(tgt_padding_mask.size())
            )
            # Forward pass, this will predict the 2nd to last token ID in every sentence
            # in actual translation, we can't pass in all in one go, and need to do 1 by 1
            # to get the auto-regressive symbol for next in sequence.
            # but in this case, we are using teacher forcing, thus we already "know" the
            # next sequence, thus able to pass in a parallel vector
            logits = model(
                src = src,
                tgt = tgt_input,
                src_mask = src_mask,
                tgt_mask = tgt_mask,
                src_padding_mask = src_padding_mask,
                tgt_padding_mask = tgt_padding_mask,
                memory_key_padding_mask = src_padding_mask,
            )
            self.logger.debug(
                'Done forward pass epoch no ' + str(epoch_no) + ', step #' + str(step)
                + ' with teacher forcing through model "' + str(type(model))
                + '" produced logits of shape ' + str(logits.size()) + ' from source shape ' + str(src.size())
                + ', target shape ' + str(tgt_input.size()) + ', source mask shape ' + str(src_mask.size())
                + ', target mask shape ' + str(tgt_mask.size()) + ', source padding mask shape '
                + str(src_padding_mask.size()) + ', target padding mask shape ' + str(tgt_padding_mask.size())
            )

            # same as the logits prediction, take from the 2nd token onwards, skip the first token ID for every sentence
            tgt_out = tgt[1:, :]
            self.logger.debug('tgt_out is from 2nd token onwards: ' + str(tgt_out))
            self.logger.debug('logits: ' + str(logits))
            # logits flattenned to 2D, and tgt_out flattenned to a 1D vector
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss = loss / accum_steps
            losses += loss.item()

            if mode == 'train':
                loss.backward()
                if done_accum:
                    # adjust model parameter weights
                    optimizer.step()

            self.logger.debug(
                'Epoch #' + str(epoch_no) + ', step #' +  str(step) +  ' src/tgt len ' + str(len(src))
                + ' loss =' + str(losses) + '. Src size ' + str(src.size()) + '\n' + str(src)
                + '\n tgt size ' + str(tgt.size())+ '\n' + str(tgt)
            )

        return losses / len(list(dataset_dataloader))

    def train_small_gpu_footprint(
            self,
            rounds,
            epochs = 18,
            # if out of memory, reduce batch size
            batch_size = 128,
            accum_steps = 1,
            # usually is limited by GPU 16G memory, means we have to train multiple rounds by saving model params
            max_samples = 20000,
            # permitted values 'state_dict', 'serializable'
            # saving as "state_dict" allows to reload from any class, saving as "serializable" need exact namespace
            save_as = SAVE_FTYPE_STATE_DICT,
            round_test_texts = (),
    ):
        for i in range(rounds):
            self.logger.info('Starting training small footprint round ' + str(i))
            self.train(
                epochs = epochs,
                batch_size = batch_size,
                accum_steps = accum_steps,
                max_samples = max_samples,
                save_as = save_as,
            )
            self.logger.info('Done training small footprint round ' + str(i))
            self.load_old_state = True
            # if i < rounds-1:
            #     self.load_all()
            if round_test_texts:
                self.logger.info('Test texts for round ' + str(i))
                for txt in round_test_texts:
                    self.logger.info(
                        '\n  "' + str(txt) + '"\n    --> "' + str(self.translate(self.nn_transformer, txt)) + '"'
                    )

    def train(
            self,
            epochs = 18,
            # if out of memory, reduce batch size
            batch_size = 128,
            accum_steps = 1,
            # usually is limited by GPU 16G memory, means we have to train multiple rounds by saving model params
            max_samples = 20000,
            # permitted values 'state_dict', 'serializable'
            # saving as "state_dict" allows to reload from any class, saving as "serializable" need exact namespace
            save_as = SAVE_FTYPE_STATE_DICT,
    ):
        if self.device == "cuda":
            self.logger.info('Clearing cuda cache...')
            torch.cuda.empty_cache()
            gc.collect()

        # create model directory for saving model if not exist first
        model_filepath = self.get_save_filepath()

        data_loader_trn, data_loader_val = self.get_dataloader(batch_size=batch_size, max_samples=max_samples)
        self.logger.info(
            'Data loader train length ' + str(len(data_loader_trn)) + ', validation length ' + str(len(data_loader_val))
        )

        self.logger.info(
            'Training started with device "' + str(self.device) + '", emb size ' + str(self.emb_size)
            + ', total epochs ' + str(epochs) + ', batch size ' + str(batch_size)
            + ', gradient accumulation steps ' + str(accum_steps) + ', max samples ' + str(max_samples)
        )
        for epoch in range(1, epochs + 1):
            start_time = timer()
            train_loss = self.evaluate(
                mode  = 'train',
                model = self.nn_transformer,
                dataset_dataloader = data_loader_trn,
                epoch_no  = epoch,
                optimizer = self.optimizer,
                accum_steps = accum_steps,
            )
            end_time = timer()

            self.logger.debug('Starting evaluation of validation loss for epoch #' + str(epoch))
            val_loss = self.evaluate(
                mode  = 'validate',
                model = self.nn_transformer,
                dataset_dataloader = data_loader_val,
                epoch_no  = epoch,
                optimizer = None,
                accum_steps = 1,
            )
            self.logger.info((
                f"Device {self.device}, Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f},"
                f""f"Epoch time (train only) = {(end_time - start_time):.3f}s"
            ))

        if save_as == self.SAVE_FTYPE_STATE_DICT:
            state = {
                'model_state_dict': self.nn_transformer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(obj=state, f=model_filepath)
            self.logger.info('Saved state dicts for model/optimizer to "' + str(model_filepath) + '"')
        else:
            torch.save(obj=self.nn_transformer, f=model_filepath)
            self.logger.info('Saved serializable model to "' + str(model_filepath) + '"')

        return

    def __load_state(
            self,
            load_as = SAVE_FTYPE_STATE_DICT,
    ):
        model_filepath = self.get_save_filepath()

        if load_as == self.SAVE_FTYPE_STATE_DICT:
            state = torch.load(
                f = model_filepath,
                map_location = self.device,
            )
            self.nn_transformer.load_state_dict(
                state_dict = state['model_state_dict'],
            )
            self.__init_optimizer()
            self.optimizer.load_state_dict(
                state_dict = state['optimizer_state_dict']
            )
            self.logger.info('Loaded state dicts for model/optimizer from "' + str(model_filepath) + '"')
        else:
            raise Exception('No longer supported')
            # self.nn_transformer = torch.load(
            #     f = model_filepath,
            #     # in case the file was trained on another device like GPU, we need to map it to the current device
            #     map_location = self.device,
            # )
            # self.logger.info('Loaded serializable model from "' + str(model_filepath) + '"')
            # return

    # TODO use beam instead
    def greedy_decode(
            self,
            model,
            src,
            src_mask,
            max_len,
            start_symbol
    ):
        self.profiler.start_time_profiling(id='greedy_decode')

        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        # get context
        memory = model.encode(src=src, src_mask=src_mask)
        self.logger.debug('Encoded input tensor as context of shape ' + str(memory.size()) + '.')
        self.profiler.record_time_profiling(id='greedy_decode', msg='Model encode', logmsg=True)

        # Sentence of token IDs start with a <BOS> symbol
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        self.logger.debug('ys: ' + str(ys))
        for i in range(max_len - 1):
            memory = memory.to(self.device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(self.device)
            # decode context
            out = model.decode(tgt=ys, memory=memory, tgt_mask=tgt_mask)
            self.logger.debug(
                'Decoded context tensor as output of shape ' + str(out.size()) + ', using mask:\n' + str(tgt_mask)
            )
            # Transpose or exchange the first 2 dim, so that batch dim is moved to the left front
            # at index 0, instead of at index 1)
            out = out.transpose(0, 1)
            # generator that we defined in Seq2SeqTransformer as torch.nn.modules.linear.Linear type
            # with in_features=<embedding_size>, out_features=<vocab_size>
            # to map embedding back to token ID
            prob = model.generator(out[:, -1])
            # value/index, value we don't need
            max_prob, next_token_id = torch.max(prob, dim=1)
            self.logger.debug(
                'Probs tensor shape ' + str(prob.size()) + ', max prob ' + str(max_prob)
                + ', token id ' + str(next_token_id)
            )
            next_token_id = next_token_id.item()

            # Append word/token id to form sentence of token IDs, one token id at a time
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_token_id)], dim=0)
            self.logger.debug('ys now shape ' + str(ys.size()) + ': ' + str(ys))
            if next_token_id == self.tokenizer.get_sep_token_id():
                break

        self.profiler.record_time_profiling(id='greedy_decode', msg='Greedy decode & model decode', logmsg=True)
        return ys

    """
    During training, we need a subsequent word mask that will prevent model to look into the future words
    when making predictions. We will also need masks to hide source and target padding tokens.
    Example of a sz=4 mask
    tensor([[0., -inf, -inf, -inf],
            [0.,   0., -inf, -inf],
            [0.,   0.,   0., -inf],
            [0.,   0.,   0.,   0.]])
    """
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
    ):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(sz=tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        pad_idx = self.tokenizer.get_pad_token_id()
        src_padding_mask = (src == pad_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    # actual function to translate input sentence into target language
    def translate(
            self,
            model: torch.nn.Module,
            src_sentence: str
    ):
        self.profiler.start_time_profiling(id='translate')
        model.eval()
        if self.test_mode:
            src = src_sentence
        else:
            # src = self.text_transforms[self.lang_src].transform_text_to_tensor(src_sentence).view(-1, 1)
            src = self.tokenizer.encode(
                text = src_sentence,
                return_len = self.input_token_max_len,
                return_tensor = 'pt',
            ).view(-1, 1)
            self.profiler.record_time_profiling(id='translate', msg='Transform text to tensor', logmsg=True)

        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(
            model = model,
            src = src,
            src_mask = src_mask,
            # TODO Is 5 too conservative?
            max_len = num_tokens + 5,
            start_symbol = self.tokenizer.get_cls_token_id(),
        ).flatten()
        tgt_tokens_np = tgt_tokens.cpu().numpy()
        self.logger.info('Source token ids: ' + str(src) + ', Target tokens: ' + str(tgt_tokens))
        self.profiler.record_time_profiling(id='translate', msg='Greedy decode', logmsg=True)
        if self.test_mode:
            return tgt_tokens
        else:
            res = self.tokenizer.decode(token_ids=tgt_tokens_np.tolist(), include_special_tokens=True)
            # res = " ".join(self.text_transforms[self.lang_tgt].vocab
            #                .lookup_tokens(id_list=list(tgt_tokens.cpu().numpy()))).replace("<bos>","").replace("<eos>", "")
            return res.strip()


if __name__ == '__main__':
    logr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    er = Env(logger=logr)

    # e.g. train=1 emb=64 epoch=10 batch=16 accum=8 maxsp=5000 prevstate=1
    params = CmdLine.read_cmdline_params()
    logr.info('Cmdline params: ' + str(params))

    lang_src = 'syl'
    lang_tgt = 'bn'
    do_train = False # True if params.get('train', '1') in ['1', 'yes', 'y'] else False
    load_prev_state = True # True if params.get('prevstate', '0').strip().lower() in ('1', 'yes', 'y',) else False
    emb_size = int(params.get('emb', 16))
    # if not do_train:
    #     load_prev_state = True

    # In the form
    # [(   0, 'অ', 'ওহে'), (   1, 'অইছে', 'হয়েছে'), (   2, 'অইতল বইতল', 'মর্যাদাহীন') ... ]
    text_pairs = pd.read_csv(
        filepath_or_buffer = er.REPO_DIR + '/data/datasets/latest/sylheti-bangla.csv',
    ).to_records()
    text_pairs = [(rec[1], rec[2]) for rec in text_pairs]
    logr.info(text_pairs[:10])

    # corpora_search_folder = er.NLP_DATASET_PARACRAWL_DIR if do_train else None
    test_mode = False
    round_test_texts = [
        'Китай модернизирует армию с упором на кибервойну',
        'США вовлечены в космическую гонку с Китаем',
        'рассматривает Пекин как «угрозу растущего влияния»',
        # Bengali (Sylheti)
        'অইল ভাত', # Cooked rice (পাক হয়েছে যে ভাত)
        'অকতে জরুর', # Absolutely necessary (নিতান্ত  প্রয়োজনীয়)
        'আউয়া যাওয়া', # Stupid, senseless (বোকা, অবোধ)
    ] if not test_mode else []
    logr.info(
        'Do train: ' + str(do_train) + ', load prev state: ' + str(load_prev_state) + ', test mode: ' + str(test_mode)
    )

    trainer = TranslateSeq2SeqTrfm(
        lang_src   = lang_src,
        lang_tgt   = lang_tgt,
        input_token_max_len = 8,
        output_token_max_len = 8,
        tokenizer_model_path = './syl-bgl.tokenizer',
        load_old_state = load_prev_state,
        cache_dir  = er.MODELS_TRAINING_DIR,
        embed_size = emb_size,
        text_pairs = text_pairs,
        test_mode = test_mode,
        logger     = logr,
    )
    if do_train:
        ep = int(params.get('epoch', 10))
        bs = int(params.get('batch', 8))
        accum = int(params.get('accum', 1))
        maxsp = int(params.get('maxsp', 100000))
        logr.info(
            'Using total epochs ' + str(ep) + ', batch size ' + str(bs) + ', gradient accumulation steps ' + str(accum)
        )
        trainer.train_small_gpu_footprint(
            rounds = 100 if not test_mode else 10,
            epochs=ep, batch_size=bs, accum_steps=accum, max_samples=maxsp,
            round_test_texts = round_test_texts,
        )

    test_texts = [
        'Китай модернизирует армию с упором на кибервойну',
        'США вовлечены в космическую гонку с Китаем',
        'рассматривает Пекин как «угрозу растущего влияния»',
        'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche',
        'Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.',
        'Eine Gruppe von Menschen steht vor einem Iglu .',
        'Training without GPU on CPU is too slow',
        'The men are drinking soju in the restaurant',
        'This is a translation from English to Korean',
    ] if not test_mode else [torch.IntTensor([[55 , 77, 22, 99, 44]]).transpose(0, 1)]
    for txt in test_texts:
        logr.info('Testing text ' + str(txt))
        txt_tl = trainer.translate(trainer.nn_transformer, txt)
        logr.info('Translation "' + str(txt) + '" --> "' + str(txt_tl) + '"')
        # break

    exit(0)
