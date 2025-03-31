import logging
import torch
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from fitxf import FitUtils
from fitxf.utils import Logging


class Regression(torch.nn.Module):

    def __init__(
            self,
            # by default is a cubic polynomial
            polynomial_order: int = 3,
            learning_rate: float = 0.001,
            logger: Logging = None,
    ):
        super().__init__()
        self.polynomial_order = polynomial_order
        self.learning_rate = learning_rate
        self.logger = logger if logger is not None else logging.getLogger()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_network_arc()
        return

    def fit_linear_regression(self, X: np.ndarray, y: np.ndarray):
        rg = LinearRegression().fit(X, y)
        coef = rg.coef_
        intercept = rg.intercept_
        return coef, intercept

    def predict_linear_regression(self, X: np.ndarray, coef: np.ndarray, intercept: np.ndarray):
        return intercept + coef * X

    def init_network_arc(self):
        self.layer_1 = torch.nn.Linear(in_features=1+self.polynomial_order, out_features=1)

        # Random initialization of model parameters if not loading from a previous state
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.learning_rate,
            betas = (0.9, 0.98),
            eps = 1e-9
        )
        return

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
        self.logger.info('Creating dataloader batch size ' + str(batch_size) + ', eval percent ' + str(eval_percent))
        if y_num_classes is not None:
            y_onehot_or_value = torch.nn.functional.one_hot(
                y.to(torch.int64),
                num_classes = y_num_classes,
            ).to(torch.float)
            self.logger.info('Converted y to onehot ')
        else:
            y_onehot_or_value = y

        dataloader_train = []
        idx_batch = 0
        while True:
            j = idx_batch + batch_size
            if j-1 > len(X):
                break
            X_batch = X[idx_batch:j]
            self.logger.info('X_batch up to ' + str(j) + ': ' + str(X_batch))
            assert len(X_batch) > 0
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

    # func: function callback f(X) -> y
    def fit(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            is_categorical: bool = False,
            # if None, means we don't convert to onehot (possibly caller already done that, or not required)
            num_categories: int = None,
            # the smaller the batch size, the smaller the losses will be during training
            batch_size: int = 32,
            epochs: int = 100,
            eval_percent: float = 0.2,
            # important to prevent over-fitting
            regularization_type = None,
    ):
        fit_utils = FitUtils(logger=self.logger)

        dl_trn, dl_val, n_cutoff_train = self.create_dataloader(
            X = X,
            y = y,
            y_num_classes = num_categories,
            batch_size = batch_size,
            eval_percent = eval_percent,
            include_attn_mask = False,
        )
        self.logger.info(
            'Train length = ' + str(len(dl_trn)) + ', eval length = ' + str(len(dl_val))
            + ', cutoff train = ' +  str(n_cutoff_train)
        )
        self.logger.info('Data loader train ' + str(dl_trn))
        # raise Exception('asdf')

        losses = fit_utils.torch_train(
            model = self,
            train_dataloader = dl_trn,
            loss_func = self.loss_func,
            optimizer = self.optimizer,
            regularization_type = regularization_type,
            epochs = epochs,
        )
        self.logger.info('Train losses: ' + str(losses))
        # Important! Set back to eval mode, so subsequent forwards won't affect any weights or gradients
        self.eval()

        if eval_percent > 0:
            eval_accuracy = self.evaluate_accuracy(
                X_eval = X[n_cutoff_train:],
                y_eval = y[n_cutoff_train:],
                is_categorical = is_categorical,
                num_categories = num_categories,
            )
        else:
            eval_accuracy = None

        return {
            'eval_accuracy': eval_accuracy,
            'losses': losses,
            'dataloader_train': dl_trn,
            'dataloader_eval': dl_val,
            'index_cutoff_train': n_cutoff_train,
        }

    def forward(
            self,
            X: torch.Tensor,
    ):
        return self.layer_1(X)

    def predict(
            self,
            X: torch.Tensor,
    ):
        return self(X)


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    r = Regression(logger=lgr)
    X = np.array([[80], [65], [50], [30], [10]])
    y = np.array([6, 5, 4, 3, 2])
    cf, ic = r.fit_linear_regression(X=X, y=y)
    lgr.info('Coef ' + str(cf) + ', intercept ' + str(ic))
    y_ = r.predict_linear_regression(X=X, coef=cf, intercept=ic)
    lgr.info(y_)

    order = 2
    r_poly = Regression(
        polynomial_order = order,
        learning_rate = 0.01,
        logger = lgr,
    )
    X_poly = torch.Tensor([
        [x[0]**i for i in range(order+1)] for x in X
    ])
    y_poly = torch.from_numpy(y).to(torch.float)
    lgr.info(X_poly)
    r_poly.fit(
        X = X_poly,
        y = y_poly,
        eval_percent = 0.,
        batch_size = len(X_poly),
        epochs = 100,
        # regularization_type = "L1",
    )
    lgr.info('Model parameters ' + str(r_poly.state_dict()))
    lgr.info('y predict ' + str(r_poly(X_poly)))

    exit(0)
