import logging
import torch
import math
from fitxf.math.utils.Logging import Logging


class ClassifierArcInterface:

    def __init__(
            self,
            # old state
            model_filepath: str = None,
            in_features: int = None,
            out_features: int = None,
            n_hidden_features: int = 100,
            hidden_functions: list = (torch.nn.Linear, torch.nn.Linear, torch.nn.Linear),
            activation_functions: list = (torch.nn.ReLU, torch.nn.ReLU, torch.nn.Softmax),
            loss_function = torch.nn.CrossEntropyLoss,
            dropout_rate: float = 0.2,
            learning_rate: float = 0.0001,
            logger = None,
    ):
        self.model_filepath = model_filepath
        self.in_features = in_features
        self.out_features = out_features
        self.n_hidden_features = n_hidden_features if type(n_hidden_features) in [list, tuple] \
            else [n_hidden_features, int(round(n_hidden_features / 2))]
        self.hidden_functions = hidden_functions
        self.activation_functions = activation_functions
        self.loss_function = loss_function
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
            # if None, means we don't convert to onehot (possibly caller already done that, or not required)
            y_num_classes: int = None,
            batch_size: int = 32,
            eval_percent: float = 0.2,
            include_attn_mask: bool = False,
    ):
        if y_num_classes is not None:
            y_onehot_or_value = torch.nn.functional.one_hot(
                y.to(torch.int64),
                num_classes = y_num_classes,
            ).to(torch.float)
        else:
            y_onehot_or_value = y

        dataloader_train = []
        idx_batch = 0
        while True:
            j = idx_batch + batch_size
            if j-1 > len(X):
                break
            X_batch = X[idx_batch:j]
            y_onehot_batch = y_onehot_or_value[idx_batch:j]
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

    def to_onehot(
            self,
            y: torch.Tensor,
    ):
        n_values = torch.max(y.to(torch.int)) + 1
        onehot = torch.eye(n_values)[y.to(torch.int)]
        onehot_f = onehot.to(torch.float)
        return onehot_f
        # alternatively, can also use the torch builtin function
        # return torch.nn.functional.one_hot(
        #     y.to(torch.int64),
        #     num_classes = y_num_classes,
        # ).to(torch.float)

    def forward(
            self,
            x: torch.Tensor,
    ):
        raise Exception('Must be implemented by child class')

    def fit(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            is_categorical: bool = True,
            # if None, means we don't convert to onehot (possibly caller already done that, or not required)
            num_categories: int = None,
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


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    exit(0)
