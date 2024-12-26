import logging
import torch
from fitxf.math.fit.arc.ClassifierArcInterface import ClassifierArcInterface
from fitxf.math.fit.arc.ClassifierArcUnitTest import ClassifierArcUnitTest
from fitxf import FitUtils
from fitxf.math.utils.Logging import Logging


class ClassifierArc(torch.nn.Module, ClassifierArcInterface):

    # Problem with using dynamic array to store layers is that we can't load back from file (torch limitation)
    USE_ARRAY = False

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
        torch.nn.Module.__init__(self)
        ClassifierArcInterface.__init__(
            self = self,
            in_features = in_features,
            out_features = out_features,
            n_hidden_features = n_hidden_features,
            activation_functions = activation_functions,
            dropout_rate = dropout_rate,
            learning_rate = learning_rate,
            logger = logger,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.n_hidden_features = n_hidden_features if type(n_hidden_features) is list else\
            [n_hidden_features, int(round(n_hidden_features / 2))]
        self.activation_functions = activation_functions
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.logger = logger if logger is not None else logging.getLogger()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__init_neural_network_arc()
        return

    def __init_neural_network_arc(
            self,
    ):
        if self.USE_ARRAY:
            # Hidden layer to auto discover features
            n_all = [self.in_features] + self.n_hidden_features
            self.config_hidden_layers = [
                {'in': n_all[i], 'out': n_all[i+1], 'act': self.activation_functions[i]}
                for i in range(len(self.n_hidden_features))
            ]
            self.layers_hidden = []
            n_out_last = self.config_hidden_layers[-1]["out"]

            for hl in self.config_hidden_layers:
                in_f, out_f, act_func = hl['in'], hl['out'], hl['act']
                self.layers_hidden.append(torch.nn.Linear(in_features=in_f, out_features=out_f))
                if act_func is not None:
                    self.layers_hidden.append(act_func())
                    self.layers_hidden.append(torch.nn.Dropout(self.dropout_rate))
        else:
            self.layer_hidden_fc1 = torch.nn.Linear(
                in_features = self.in_features,
                out_features = self.n_hidden_features[0],
            )
            self.layer_fc1_norm = torch.nn.LayerNorm(
                normalized_shape = self.n_hidden_features[0],
            )
            if self.activation_functions[0] is not None:
                self.layer_hidden_fc1_act = self.activation_functions[0]()
                self.layer_dropout_fc1_drop = torch.nn.Dropout(p=self.dropout_rate)

            n_out_last = self.n_hidden_features[1]
            self.layer_hidden_fc2 = torch.nn.Linear(
                in_features = self.n_hidden_features[1-1],
                out_features = n_out_last,
            )
            self.layer_fc2_norm = torch.nn.LayerNorm(
                normalized_shape = self.n_hidden_features[1],
            )
            if self.activation_functions[1] is not None:
                self.layer_hidden_fc2_act = self.activation_functions[1]()
                self.layer_dropout_fc2_drop = torch.nn.Dropout(p=self.dropout_rate)

        self.layer_classify = torch.nn.Linear(
            in_features = n_out_last,
            out_features = self.out_features,
        )
        self.layer_softmax = torch.nn.Softmax(dim=-1)

        # Random initialization of model parameters if not loading from a previous state
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.learning_rate,
            betas = (0.9, 0.98),
            eps = 1e-9
        )
        self.logger.info('Network initialized successfully')
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
        if self.USE_ARRAY:
            h_out = x
            for layer in self.layers_hidden:
                h_out = layer(h_out)
            h2_out = h_out
        else:
            h1 = self.layer_hidden_fc1(x)
            h1_norm = self.layer_fc1_norm(h1)
            # self.logger.debug('Linear layer shape ' + str(h1.shape))
            if self.activation_functions[0] is not None:
                h1_act = self.layer_hidden_fc1_act(h1_norm)
                h1_act_drop = self.layer_dropout_fc1_drop(h1_act)
                h1_out = h1_act_drop
            else:
                h1_out = h1_norm

            h2 = self.layer_hidden_fc2(h1_out)
            h2_norm = self.layer_fc2_norm(h2)
            if self.activation_functions[1] is not None:
                h2_act = self.layer_hidden_fc2_act(h2_norm)
                h2_act_drop = self.layer_dropout_fc2_drop(h2_act)
                h2_out = h2_act_drop
            else:
                h2_out = h2_norm

        h_last = self.layer_classify(h2_out)
        # self.logger.debug('Sentiment linear layer shape ' + str(senti.shape))
        category_prob = self.layer_softmax(h_last)
        return category_prob

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
            out = self(X[n_cutoff_train:])
            self.logger.debug('Out for eval test: ' + str(out))
            out_cat = torch.argmax(out, dim=-1)
            self.logger.debug('Out categories for eval test: ' + str(out_cat))
            assert len(out) == len(out_cat)
            correct = 1 * (y[n_cutoff_train:] - out_cat == 0)
            eval_accuracy = torch.sum(correct) / len(correct)
            eval_accuracy = eval_accuracy.item()
            self.logger.info(
                'Evaluation results: Total correct ' + str(torch.sum(correct).item()) + ' from length ' + str(len(correct))
                + ', accuracy ' + str(eval_accuracy)
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
    for test in ['max', 'sum']:
        ut.test(
            test_function = test,
        )
    exit(0)
