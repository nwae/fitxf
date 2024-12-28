import logging
import numpy as np
import torch
import joblib
from fitxf.math.fit.arc.ClassifierArcInterface import ClassifierArcInterface
from fitxf.math.fit.arc.ClassifierArcUnitTest import ClassifierArcUnitTest
from sklearn.ensemble import RandomForestClassifier
from fitxf.math.utils.Logging import Logging


class ClassifierArcRf(ClassifierArcInterface):

    USE_ARRAY = False

    def __init__(
            self,
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
        super().__init__(
            model_filepath = model_filepath,
            logger = logger,
        )
        self.model_rf = None
        if self.model_filepath is not None:
            self.from_old_states(model_filepath=self.model_filepath)
        return

    def from_old_states(
            self,
            model_filepath: str,
    ):
        self.model_rf = joblib.load(model_filepath)
        self.logger.info('Loaded old state for random forest model from file "' + str(self.model_filepath) + '"')
        return self.model_rf

    def save_states(
            self,
            model_filepath: str,
            additional_info: dict,
    ):
        joblib.dump(self.model_rf, model_filepath)
        self.logger.info('Saved random forest model to "' + str(model_filepath) + '"')
        return

    def forward(
            self,
            x: torch.Tensor,
    ):
        return self.predict(X=x)

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
        n_cutoff_x_train = int(np.round((1 - eval_percent) * len(X)))
        self.logger.info(
            'X train cutoff = ' + str(n_cutoff_x_train) + ' total length = ' + str(len(X))
        )
        self.model_rf = RandomForestClassifier().fit(X[:n_cutoff_x_train], y[:n_cutoff_x_train])
        trees = self.model_rf.estimators_
        self.logger.info('Classified via random forest classifier, decision trees total ' + str(len(trees)))
        [self.logger.info(str('Tree #' + str(i) + ':' + str(dt))) for i, dt in enumerate(trees)]

        # Validate back
        out_cat, out_prob = self.predict(X[n_cutoff_x_train:])
        out_cat = out_cat.flatten()
        self.logger.info('Out cat shape ' + str(out_cat.shape) + ', y shape ' + str(y.shape))
        correct = 1 * (y[n_cutoff_x_train:] - out_cat == 0)
        eval_accuracy = torch.sum(correct) / len(correct)
        self.logger.info(
            'Evaluation results for random forest: Total correct ' + str(torch.sum(correct).item())
            + ' from length ' + str(len(correct)) + ', accuracy ' + str(eval_accuracy.item())
        )
        return {
            'eval_accuracy': eval_accuracy.item(),
            'losses': None,
            'dataloader_train': list(zip(X[:n_cutoff_x_train], y[:n_cutoff_x_train])),
            'dataloader_eval': list(zip(X[n_cutoff_x_train:], y[n_cutoff_x_train:])),
            'index_cutoff_train': n_cutoff_x_train,
        }

    def predict(
            self,
            X: torch.Tensor,
    ):
        # RF predicts only 1 category per row
        out_cat = self.model_rf.predict(X)
        out_cat_torch = torch.Tensor([[v] for v in out_cat])
        # self.logger.debug(out_cat_torch)
        out_prob_torch = torch.Tensor([[1.] for v in out_cat])
        return out_cat_torch, out_prob_torch


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    ut = ClassifierArcUnitTest(child_class=ClassifierArcRf, logger=lgr)
    ut.test()
    exit(0)
