import logging
import torch
from fitxf.math.fit.arc.ClassifierArcInterface import ClassifierArcInterface
from fitxf.math.fit.arc.ClassifierArcUnitTest import ClassifierArcUnitTest
from fitxf import FitUtils
from fitxf.math.utils.Logging import Logging


class ClassifierArc(torch.nn.Module, ClassifierArcInterface):

    # Problem with using dynamic array to store layers is that we can't load back from file (torch limitation)
    USE_ARRAY = False

    ACT_FUNCS = (
        torch.nn.Tanh, torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.LogSigmoid, torch.nn.Softmax,
    )

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
        torch.nn.Module.__init__(self)
        ClassifierArcInterface.__init__(
            self = self,
            model_filepath = model_filepath,
            in_features = in_features,
            out_features = out_features,
            n_hidden_features = n_hidden_features,
            hidden_functions = hidden_functions,
            activation_functions = activation_functions,
            loss_function = loss_function,
            dropout_rate = dropout_rate,
            learning_rate = learning_rate,
            logger = logger,
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__init_neural_network_arc()
        if self.model_filepath is not None:
            # if loading from filepath, will overwrite the above neural network params (if different)
            self.from_old_states(model_filepath=self.model_filepath)
        return

    def __init_neural_network_arc(
            self,
    ):
        assert len(self.hidden_functions) == 3, \
            'Expect 3 hidden functions but got ' + str(len(self.hidden_functions))
        assert self.hidden_functions[0] is not None

        n_out_1 = self.n_hidden_features[0]
        self.layer_hidden_fc1 = self.hidden_functions[0](
            in_features = self.in_features,
            out_features = n_out_1,
        )
        if self.activation_functions[0] in self.ACT_FUNCS:
            self.layer_hidden_fc1_act = self.activation_functions[0]()
        elif self.activation_functions[0] == torch.nn.LayerNorm:
            self.layer_hidden_fc1_act = torch.nn.LayerNorm(
                normalized_shape = n_out_1,
            )
        else:
            self.layer_hidden_fc1_act = None
            self.logger.warning(
                'Ignore non-recognized activation function "'
                + str(self.activation_functions[0]) + '"'
            )
        self.layer_dropout_fc1_drop = torch.nn.Dropout(p=self.dropout_rate)

        self.layer_hidden_fc2 = None
        self.layer_hidden_fc2_act = None
        self.layer_dropout_fc2_drop = None
        if self.hidden_functions[1] is not None:
            n_out_2 = self.n_hidden_features[1]
            self.layer_hidden_fc2 = self.hidden_functions[1](
                in_features = n_out_1,
                out_features = n_out_2,
            )
            if self.activation_functions[1] in self.ACT_FUNCS:
                self.layer_hidden_fc2_act = self.activation_functions[1]()
            elif self.activation_functions[1] == torch.nn.LayerNorm:
                self.layer_hidden_fc2_act = torch.nn.LayerNorm(
                    normalized_shape = n_out_2,
                )
            else:
                self.logger.warning(
                    'Ignore non-recognized activation function "'
                    + str(self.activation_functions[0]) + '"'
                )
            self.layer_dropout_fc2_drop = torch.nn.Dropout(p=self.dropout_rate)
        else:
            n_out_2 = n_out_1

        self.layer_hidden_last = None
        self.layer_act_last = None
        if self.hidden_functions[2] is not None:
            self.layer_hidden_last = torch.nn.Linear(
                in_features = n_out_2,
                out_features = self.out_features,
            )
            if self.activation_functions[2] in self.ACT_FUNCS:
                self.layer_act_last = self.activation_functions[2]()
            elif self.activation_functions[2] == torch.nn.LayerNorm:
                self.layer_act_last = torch.nn.LayerNorm(
                    normalized_shape = self.out_features,
                )
            else:
                self.logger.warning(
                    'Ignore non-recognized activation function "'
                    + str(self.activation_functions[2]) + '"'
                )

        # Random initialization of model parameters if not loading from a previous state
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.loss_func = self.loss_function()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.learning_rate,
            betas = (0.9, 0.98),
            eps = 1e-9
        )
        self.logger.info('Network initialized successfully ' + str(self))
        return

    def from_old_states(
            self,
            model_filepath: str,
    ):
        state = torch.load(
            f = model_filepath,
            map_location = self.device,
        )
        model_state_dict = state['model_state_dict']
        optimizer_state_dict = state['optimizer_state_dict']
        self.load_state_dict(
            state_dict = model_state_dict,
        )
        self.optimizer.load_state_dict(
            state_dict = optimizer_state_dict
        )
        self.logger.info('Loaded old state from file "' + str(model_filepath) + '"')
        return state

    def save_states(
            self,
            model_filepath: str,
            additional_info: dict,
    ):
        state = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        state.update(additional_info)
        torch.save(obj=state, f=model_filepath)
        self.logger.info('Saved state dicts for model/optimizer to "' + str(model_filepath) + '"')

    def forward(
            self,
            x: torch.Tensor,
    ):
        h1 = self.layer_hidden_fc1(x)
        # self.logger.debug('Linear layer shape ' + str(h1.shape))
        if self.layer_hidden_fc1_act is not None:
            h1_act_or_norm = self.layer_hidden_fc1_act(h1)
        else:
            h1_act_or_norm = h1
        h1_out = self.layer_dropout_fc1_drop(h1_act_or_norm)

        if self.layer_hidden_fc2 is not None:
            h2 = self.layer_hidden_fc2(h1_out)
            if self.layer_hidden_fc2_act is not None:
                h2_act_or_norm = self.layer_hidden_fc2_act(h2)
            else:
                h2_act_or_norm = h2
            h2_out = self.layer_dropout_fc2_drop(h2_act_or_norm)
        else:
            h2_out = h1_out

        if self.layer_hidden_last is not None:
            h_last = self.layer_hidden_last(h2_out)
            # self.logger.debug('Sentiment linear layer shape ' + str(senti.shape))
            if self.layer_act_last is not None:
                last_out = self.layer_act_last(h_last)
            else:
                last_out = h_last
        else:
            last_out = h2_out
        return last_out

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
        fit_utils = FitUtils(logger=self.logger)
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

    def evaluate_accuracy(
            self,
            X_eval: torch.Tensor,
            y_eval: torch.Tensor,
            is_categorical: bool = True,
            # if None, means we don't convert to onehot (possibly caller already done that, or not required)
            num_categories: int = None,
    ):
        assert len(X_eval) == len(y_eval)
        out_sorted, _ = self.predict(X=X_eval)
        self.logger.debug(
            'Out for eval test (shape ' + str(out_sorted.shape) + ', y shape ' + str(y_eval.shape)
            + '): ' + str(out_sorted)
        )
        out_top = out_sorted[:,0]
        y_expected = y_eval

        self.logger.debug(
            'Out categories for eval test:\n' + str(list(zip(out_top.tolist(), y_expected.tolist())))
        )
        correct = 1 * (y_expected - out_top == 0)
        eval_accuracy = torch.sum(correct) / len(correct)
        eval_accuracy = eval_accuracy.item()
        self.logger.info(
            'Evaluation results: Total correct ' + str(torch.sum(correct).item()) + ' from length ' + str(len(correct))
            + ', accuracy ' + str(eval_accuracy)
        )
        return eval_accuracy

    def predict(
            self,
            X: torch.Tensor,
    ):
        self.eval()
        out = self(X)
        # self.logger.debug('Predict input X:\n' + str(X[:,:8]) + '\nOut for predict:\n' + str(out))
        out_arg_sorted = torch.argsort(out, descending=True, dim=-1)
        # out_cat = torch.argmax(out, dim=-1)
        self.logger.debug('Out arg sorted for predict: ' + str(out_arg_sorted))
        assert len(out) == len(out_arg_sorted)
        out_val_sorted = torch.Tensor([[out[i][j].item() for j in row] for i, row in enumerate(out_arg_sorted)])
        self.logger.debug('Model output inference argsort: ' + str(out_val_sorted))

        return out_arg_sorted, out_val_sorted


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    ut = ClassifierArcUnitTest(child_class=ClassifierArc, logger=lgr)
    ut.test()
    exit(0)
