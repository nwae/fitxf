import logging
from types import NoneType

import numpy as np
import torch
import os
from fitxf.math.fit.arc.ClassifierArcInterface import ClassifierArcInterface
from fitxf.math.utils.Logging import Logging


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
    ):
        accuracies = {}
        for f in [
            # 'max',
            'sum',
        ]:
            accuracies[f] = self.test_by_func(
                load_state_if_exists = False,
                test_function = f,
            )
        self.logger.info('Tests passed with accuracies ' + str(accuracies))
        return

    def test_by_func(
            self,
            load_state_if_exists = False,
            # max, sum
            test_function: str = 'max',
    ):
        X = torch.rand(size=(1024, 4))

        if test_function == 'max':
            is_categorical = True
            # category is just the largest index
            y, n_cat = torch.argmax(X, dim=-1), X.shape[-1]
            assert len(np.unique((y))) == n_cat
            # y_onehot_or_value = torch.nn.functional.one_hot(y, num_classes=n_cat).to(torch.float)
            n_hidden_features = (100, 50,)
            hidden_functions = (torch.nn.Linear, torch.nn.Linear, torch.nn.Linear)
            # for max function, it can be any function that is always increasing, thus we choose
            # Tanh() function since it is nicely bounded & satisfies always increasing
            activation_functions = (torch.nn.Tanh, torch.nn.Tanh, torch.nn.Softmax)
            loss_f = torch.nn.CrossEntropyLoss
            dropout = 0.2
            learn_rate = 0.001
            regularization_type = 0.
            num_epochs, batch_sz = 10, 16
            acc_thr = 0.90
        else:
            is_categorical = True
            # category is the sum of the rounded X
            y, n_cat = torch.sum(torch.round(X).to(torch.int), dim=-1), X.shape[-1] + 1
            assert len(np.unique((y))) == n_cat
            # y_onehot_or_value = torch.unsqueeze(y_onehot_or_value, dim=1)
            # y_onehot_or_value = torch.nn.functional.one_hot(y, num_classes=n_cat).to(torch.float)
            n_hidden_features = (100, None)
            hidden_functions = (torch.nn.Linear, None, torch.nn.Linear)
            # since summation is a linear function, any non-linear activation will cause problems
            activation_functions = (None, None, torch.nn.Softmax)
            loss_f = torch.nn.CrossEntropyLoss
            dropout = 0.
            learn_rate = 0.001
            regularization_type = 0.
            # for summation, need smaller batch sizes, otherwise can't converge
            num_epochs, batch_sz = 10, 4
            acc_thr = 0.60

        assert len(X) == len(y)
        self.logger.info(
            'X input (shape ' + str(X.shape) + '):\n' + str(X) + '\ny output (shape ' + str(y.shape)
            + '):\n' + str(y)
        )
        clf = self.child_class(
            in_features = X.shape[-1],
            out_features = n_cat,
            n_hidden_features = n_hidden_features,
            dropout_rate = dropout,
            learning_rate = learn_rate,
            hidden_functions = hidden_functions,
            activation_functions = activation_functions,
            loss_function = loss_f,
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
                is_categorical = is_categorical,
                # we already calculated onehot ourselves
                num_categories = n_cat,
                batch_size = batch_sz,
                epochs = num_epochs,
                regularization_type = regularization_type,
            )
            acc = res['eval_accuracy']
        # [self.logger.info(p) for p in clf.parameters()]
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
        self.logger.info('TEST PASSED FOR "' + str(test_function) + '"')
        return acc


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    exit(0)
