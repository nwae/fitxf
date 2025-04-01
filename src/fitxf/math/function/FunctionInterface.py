import logging
import torch
from fitxf.utils import Logging


class FunctionInterface:

    def __init__(self, params: torch.Tensor = None, logger: Logging = None):
        self.logger = logger if logger is not None else logging.getLogger()
        self.params = params
        return

    def forward(self, X: torch.Tensor, params: torch.Tensor = None) -> torch.Tensor:
        raise Exception('Must be implemented by child class!!')

    def derivative_of_params(self, X: torch.Tensor, delta: float = 0.001) -> torch.Tensor:
        if self.params is None:
            return torch.DoubleTensor([0.])

        derv = [None] * len(self.params)
        for i, _ in enumerate(self.params):
            prms_i_delta = self.params.clone().detach()
            prms_i_delta[i] = prms_i_delta[i] + delta
            # TODO need to normalize?
            f_X_d = self.forward(X=X, params=prms_i_delta)
            f_X_0 = self.forward(X=X, params=self.params)
            self.logger.debug('f(X, params+delta) = ' + str(f_X_d) + ' for X ' + str(X))
            self.logger.debug('f(X, params) = ' + str(f_X_0) + ' for X ' + str(X))
            derv_i = (f_X_d - f_X_0)
            self.logger.debug(
                'Derivative for i=' + str(i) + ', ' + str(derv_i) + ', params delta ' + str(prms_i_delta)
                + ', params ' + str(self.params)
            )
            derv[i] = derv_i.tolist()
        tsr_derv = torch.DoubleTensor(derv)
        tsr_derv = torch.transpose(tsr_derv, 0, 1)
        self.logger.info('Final derivative ' + str(tsr_derv))
        return tsr_derv


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    class F(FunctionInterface):
        def __init__(self, params: torch.Tensor = None, logger: Logging = None):
            super().__init__(params=params, logger=logger)

        def forward(self, X: torch.Tensor, params: torch.Tensor = None) -> torch.Tensor:
            prms = self.params if params is None else params
            return prms[0] + prms[1]*X + prms[2]*(X**2)

    params = torch.Tensor([2.0, 1.0, 0.5])
    X_test = torch.Tensor([[-1], [0]])
    f = F(params=params, logger=lgr)
    lgr.info(f.forward(X=X_test))
    lgr.info(f.derivative_of_params(X=X_test, delta=0.1))
    exit(0)
