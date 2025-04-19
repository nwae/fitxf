import logging
import torch
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
        self.fit_utils = FitUtils(logger=self.logger)
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
        # for polynomial a0 + a1*x + a2*x^2 + ...
        # a0 = bias or self.parameters()[0]
        # a1, a2, ... = self.parameters()[1]
        self.layer_1 = torch.nn.Linear(in_features=self.polynomial_order, out_features=1)

        # Random initialization of model parameters if not loading from a previous state
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.learning_rate,
            # betas = (0.9, 0.98),
            # eps = 1e-9
        )
        return

    def split_data_train(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            # if None, means we don't convert to onehot (possibly caller already done that, or not required)
            num_categories: int = None,
            # the smaller the batch size, the smaller the losses will be during training
            batch_size: int = 32,
            eval_percent: float = 0.2,
            include_attn_mask: bool = False,
    ):
        dl_trn, dl_val, n_cutoff_train = self.fit_utils.create_dataloader(
            X = X,
            y = y,
            y_num_classes = num_categories,
            batch_size = batch_size,
            eval_percent = eval_percent,
            include_attn_mask = include_attn_mask,
        )
        self.logger.info(
            'Train length = ' + str(len(dl_trn)) + ', eval length = ' + str(len(dl_val))
            + ', cutoff train = ' +  str(n_cutoff_train)
        )
        self.logger.info('Data loader train ' + str(dl_trn))
        # raise Exception('asdf')
        return dl_trn, dl_val, n_cutoff_train

    def fit_custom_func(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            # with interfaces forward(), parameters() & derivative()
            custom_func,
            loss_func,
            batch_size: int = 32,
            eval_percent: float = 0.2,
            epochs: int = 100,
    ):
        dl_trn, dl_val, n_cutoff_train = self.split_data_train(
            X = X,
            y = y,
            num_categories = None,
            batch_size = batch_size,
            eval_percent = eval_percent,
        )
        for i in range(epochs):
            y_round = custom_func.forward(X)
            loss_val = np.sum((y_round - y) ** 2)


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
            eval_percent: float = 0.2,
            epochs: int = 100,
            # important to prevent over-fitting
            regularization_type = None,
    ):
        dl_trn, dl_val, n_cutoff_train = self.split_data_train(
            X = X,
            y = y,
            num_categories = num_categories,
            batch_size = batch_size,
            eval_percent = eval_percent,
        )

        losses = self.fit_utils.torch_train(
            model = self,
            train_dataloader = dl_trn,
            loss_func = self.loss_func,
            optimizer = self.optimizer,
            regularization_type = regularization_type,
            epochs = epochs,
        )
        self.logger.info('Train losses: ' + str(losses))
        self.coeff_a = torch.nan
        self.coeff_a0 = torch.nan
        for i, prm in enumerate(self.parameters()):
            if i == 0:
                self.coeff_a = torch.Tensor([v.item() for v in prm[0]])
            elif i == 1:
                self.coeff_a0 = prm[0].item()
        self.logger.info('a0=' + str(self.coeff_a0) + ', a=' + str(self.coeff_a))
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
            method: str = None,
    ):
        if method is None:
            return self.layer_1(X)
        else:
            # should return the same as above, just that we calculate manually for demo purposes
            return self.coeff_a0 + torch.sum(self.coeff_a * X, dim=1)


class RegressionUnitTest:
    def __init__(self, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        self.regressor = Regression(logger=self.logger)
        self.custom_func = self.CustomFunctionExample(logger=self.logger)
        return

    class CustomLossMseExample:
        def __init__(self, logger: Logging = None):
            return

        def forward(self, y_eval: torch.Tensor, y: torch.Tensor):
            return np.sum((y_eval - y) ** 2)

    class CustomFunctionExample:
        def __init__(self, logger: Logging = None):
            self.logger = logger if logger is not None else logging.getLogger()
            self.params = torch.Tensor([2, 0.1])
            self.params_delta = torch.zeros(size=self.params.size())
            assert self.params.size() == self.params_delta.size()
            self.params_shape = self.params.size()
            return

        def get_params(self):
            return self.params

        def update_params(self, new_params: torch.Tensor):
            assert new_params.size() == self.params_shape
            self.params = new_params

        def forward(self, X: torch.Tensor, params: torch.Tensor = None):
            prms = self.params if params is None else params
            assert prms.size() == self.params_shape
            A, k = prms
            return A * torch.exp(-k * X)

        def derivative(self, X: torch.Tensor, delta: float = 0.001):
            derv = [None]*len(self.params )
            for i, _ in enumerate(self.params):
                prms_i_delta = self.params.clone().detach()
                prms_i_delta[i] = prms_i_delta[i] + delta
                # TODO need to normalize?
                derv_i = self.forward(X=X, params=prms_i_delta) - self.forward(X=X, params=self.params)
                self.logger.info(
                    'Derivative for i=' + str(i) + ', ' + str(derv_i) + ', params delta ' + str(prms_i_delta)
                    + ', params ' + str(self.params)
                )
                derv[i] = derv_i.tolist()
            tsr_derv = torch.Tensor(derv)
            self.logger.info('Final derivative ' + str(tsr_derv))
            return tsr_derv

    def test_linear_regression(self):
        X = np.array([[80], [65], [50], [30], [10]])
        y = np.array([6, 5, 4, 3, 2])
        cf, ic = self.regressor.fit_linear_regression(X=X, y=y)
        self.logger.info('Type ' + str(type(cf)) + ', ' + str(type(ic)))
        exp_cf, exp_ic = np.array([0.05681818]), 1.3295454545454541
        assert np.sum((cf - exp_cf)**2) < 0.0000000001, 'Fit coefficient ' + str(cf) + ', not expected ' + str(exp_cf)
        assert ic == exp_ic, 'Fit intercept ' + str(ic) + ' not expected ' + str(exp_ic)

        self.logger.info('Coef ' + str(cf) + ', intercept ' + str(ic))
        y_ = self.regressor.predict_linear_regression(X=X, coef=cf, intercept=ic)
        exp_y_pred = [[5.875     ], [5.02272727], [4.17045455], [3.03409091], [1.89772727]]
        assert np.sum((y_ - exp_y_pred)**2) < 0.0000000001, \
            'Predicted values ' + str(y_) + ', not expected ' + str(exp_y_pred)
        self.logger.info('Linear Regression Tests Passed')
        return

    def test(self):
        self.test_linear_regression()

        X = np.array([[80], [65], [50], [30], [10]])
        f = self.CustomFunctionExample(logger=self.logger)
        self.logger.info(f.forward(X))
        self.logger.info(f.derivative(X))

        self.logger.info('All Tests Passed')
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    RegressionUnitTest(logger=lgr).test()

    X = np.array([[80], [65], [50], [30], [10]])
    y = np.array([6, 5, 4, 3, 2])
    order = 2
    r_poly = Regression(
        polynomial_order = order,
        learning_rate = 0.01,
        logger = lgr,
    )
    X_poly = torch.Tensor([
        [x[0]**i for i in range(1, order+1, 1)] for x in X
    ])
    y_poly = torch.from_numpy(y).to(torch.float)
    lgr.info(X_poly)
    r_poly.fit(
        X = X_poly,
        y = y_poly,
        eval_percent = 0.,
        batch_size = len(X_poly),
        epochs = 500,
        # regularization_type = "L1",
    )
    lgr.info('Model parameters ' + str(r_poly.state_dict()))
    lgr.info('y predict:\n' + str(r_poly(X_poly)))
    lgr.info('y predict manual:\n' + str(r_poly.forward(X_poly, method='manual')))
    lgr.info('Model ' + str(r_poly))

    exit(0)
