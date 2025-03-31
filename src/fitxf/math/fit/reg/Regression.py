import logging
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from fitxf import FitUtils
from fitxf.utils import Logging


class Regression(torch.nn.Module):

    def __init__(
            self,
            custom_func,
            logger: Logging = None,
    ):
        super().__init__()
        self.custom_func = custom_func
        self.logger = logger if logger is not None else logging.getLogger()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return

    def fit_linear_regression(self, X: np.ndarray, y: np.ndarray):
        rg = LinearRegression().fit(X, y)
        coef = rg.coef_
        intercept = rg.intercept_
        return coef, intercept

    def predict_linear_regression(self, X: np.ndarray, coef: np.ndarray, intercept: np.ndarray):
        return intercept + coef * X

    # func: function callback f(X) -> y
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
            learning_rate: float = 0.001,
    ):
        fit_utils = FitUtils(logger=self.logger)

        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = learning_rate,
            betas = (0.9, 0.98),
            eps = 1e-9
        )

        dl_trn, dl_val, n_cutoff_train = self.create_dataloader(
            X = X,
            y = y,
            y_num_classes = num_categories,
            batch_size = batch_size,
            eval_percent = eval_percent,
            include_attn_mask = True,
        )
        self.logger.info(
            'Train length = ' + str(len(dl_trn)) + ', eval length = ' + str(len(dl_val))
            + ', cutoff train = ' +  str(n_cutoff_train)
        )

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
        return self.custom_func(X)


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    r = Regression(custom_func=None, logger=lgr)
    X = np.array([[80], [65], [50], [30], [10]])
    y = np.array([6, 5, 4, 3, 2])
    cf, ic = r.fit_linear_regression(X=X, y=y)
    lgr.info('Coef ' + str(cf) + ', intercept ' + str(ic))
    y_ = r.predict_linear_regression(X=X, coef=cf, intercept=ic)
    lgr.info(y_)

    exit(0)
