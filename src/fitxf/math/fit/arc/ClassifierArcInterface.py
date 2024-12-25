import logging
import torch
import math
import os
from fitxf.math.utils.Logging import Logging


class ClassifierArcInterface:

    def __init__(
            self,
            in_features: int = None,
            out_features: int = None,
            n_hidden_features: int = 100,
            activation_functions: list = (torch.nn.ReLU, torch.nn.ReLU),
            dropout_rate: float = 0.2,
            learning_rate: float = 0.0001,
            logger = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.n_hidden_features = n_hidden_features if type(n_hidden_features) is list else\
            [n_hidden_features, int(round(n_hidden_features / 2))]
        self.activation_functions = activation_functions
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def from_old_states(
            self,
            model_filepath: str,
    ):
        raise Exception('Must be implemented by child class')

    def save_states(
            self,
            model_filepath: str,
            additional_info: dict,
    ):
        raise Exception('Must be implemented by child class')

    # just an array of batch tuples
    #    e.g. [(X_batch_1, y_batch_1), (X_batch_2, y_batch_2), ...]
    # or if attention masks included
    #    e.g. [(X_batch_1, attn_mask_batch_1, y_batch_1), (X_batch_2, attn_mask_batch_2, y_batch_2), ...]
    def create_dataloader(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            y_num_classes: int,
            batch_size: int = 32,
            eval_percent: float = 0.2,
            include_attn_mask: bool = False,
    ):
        y_onehot = torch.nn.functional.one_hot(
            y.to(torch.int64),
            num_classes = y_num_classes,
        ).to(torch.float)
        dataloader_train = []
        idx_batch = 0
        while True:
            j = idx_batch + batch_size
            if j-1 > len(X):
                break
            X_batch = X[idx_batch:j]
            y_onehot_batch = y_onehot[idx_batch:j]
            if include_attn_mask:
                attn_mask = torch.ones(size=X_batch.shape)
                dataloader_train.append((X_batch, attn_mask, y_onehot_batch))
            else:
                dataloader_train.append((X_batch, y_onehot_batch))
            idx_batch = j

        cutoff_batch_idx = math.floor(len(dataloader_train) * (1 - eval_percent))
        dataloader_eval = dataloader_train[cutoff_batch_idx:]
        dataloader_train = dataloader_train[:cutoff_batch_idx]
        self.logger.info(
            'Created data loader (train/eval) of length ' + str(len(dataloader_train))
            + ' / ' + str(len(dataloader_eval)) + ' of batch sizes ' + str(batch_size)
        )

        # record where we cut off train/eval
        trn_eval_cutoff_idx = cutoff_batch_idx * batch_size
        return dataloader_train, dataloader_eval, trn_eval_cutoff_idx

    def forward(
            self,
            x: torch.Tensor,
    ):
        raise Exception('Must be implemented by child class')

    def fit(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            num_categories: int,
            # the smaller the batch size, the smaller the losses will be during training
            batch_size: int = 32,
            epochs: int = 100,
            eval_percent: float = 0.2,
            # important to prevent over-fitting
            regularization_type = "L2",
    ):
        raise Exception('Must be implemented by child class')

    def predict(
            self,
            X: torch.Tensor,
    ):
        raise Exception('Must be implemented by child class')


class ClassifierArcUnitTest:
    def __init__(
            self,
            child_class: type(ClassifierArcInterface),
            logger = None,
    ):
        self.child_class = child_class
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(
            self,
            load_state_if_exists = False,
            # max, sum
            test_function: str = 'max',
    ):
        X = torch.rand(size=(1024, 4))

        if test_function == 'max':
            # category is just the largest index
            y, n_cat = torch.argmax(X, dim=-1), X.shape[-1]
            # for max function, it can be any function that is always increasing, thus we choose
            # Tanh() function since it is nicely bounded & satisfies always increasing
            activation_functions = (torch.nn.Tanh, torch.nn.Tanh)
            dropout = 0.2
            learn_rate = 0.001
            regularization_type = 0
            acc_thr = 0.8
        else:
            # category is the sum of the rounded X
            y, n_cat = torch.sum(torch.round(X), dim=-1), X.shape[-1] + 1
            # since summation is a linear function, any non-linear activation will cause problems
            activation_functions = (None, None)
            dropout = 0.
            learn_rate = 0.001
            regularization_type = 0
            acc_thr = 0.5

        assert len(X) == len(y)
        clf = self.child_class(
            in_features = X.shape[-1],
            out_features = n_cat,
            n_hidden_features = 100,
            dropout_rate = dropout,
            learning_rate = learn_rate,
            # since we are testing linear functions that depend on the value of the input,
            # don't put an activation layer
            activation_functions = activation_functions,
            logger = self.logger,
        )
        model_filepath = 'ut.' + str(self.child_class.__name__) + '.bin'

        if load_state_if_exists and os.path.exists(model_filepath):
            clf.from_old_states(
                model_filepath = model_filepath,
            )
            out_args, out_vals = clf.predict(X=X)
            # take 1st value of every row
            out_cat_top = out_args[:,:1].flatten()
            self.logger.info('Categories top predicted: ' + str(out_cat_top) + ', shape ' + str(out_cat_top.shape))
            self.logger.info('y: ' + str(y) + ', shape ' + str(y.shape))
            correct = 1 * (y - out_cat_top == 0)
            self.logger.info('Correct: ' + str(correct))
            acc = torch.sum(correct) / len(correct)
            self.logger.info(
                'Evaluation results for "' + str(self.child_class.__name__) + '": Total correct '
                + str(torch.sum(correct).item()) + ' from length ' + str(len(correct)) + ', accuracy ' + str(acc.item())
            )
        else:
            res = clf.fit(
                X = X,
                y = y,
                num_categories = n_cat,
                batch_size = 16,
                epochs = 10,
                regularization_type = regularization_type,
            )
            acc = res['eval_accuracy']
        self.logger.info(
            'Child class "' + str(self.child_class.__name__) + '" accuracy ' + str(acc)
            + ' for test "' + str(test_function) + '"'
        )
        assert acc > acc_thr, \
            'Child class "' + str(self.child_class.__name__) + '" Accuracy from evaluation ' \
            + str(acc) + ' not > ' + str(acc_thr) + ' for test "' + str(test_function) + '"'

        if not os.path.exists(model_filepath):
            clf.save_states(
                model_filepath = model_filepath,
                additional_info = {},
            )
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    exit(0)
