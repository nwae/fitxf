import logging
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
        for f in ['max', 'sum']:
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
            # category is just the largest index
            y, n_cat = torch.argmax(X, dim=-1), X.shape[-1]
            # for max function, it can be any function that is always increasing, thus we choose
            # Tanh() function since it is nicely bounded & satisfies always increasing
            activation_functions = (torch.nn.Tanh, torch.nn.Tanh)
            dropout = 0.2
            learn_rate = 0.001
            regularization_type = 0.
            acc_thr = 0.90
        else:
            # category is the sum of the rounded X
            y, n_cat = torch.sum(torch.round(X), dim=-1), X.shape[-1] + 1
            # since summation is a linear function, any non-linear activation will cause problems
            activation_functions = (None, None)
            dropout = 0.
            learn_rate = 0.001
            regularization_type = 0.
            acc_thr = 0.60

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
        self.logger.info('TEST PASSED FOR "' + str(test_function) + '"')
        return acc


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    exit(0)
