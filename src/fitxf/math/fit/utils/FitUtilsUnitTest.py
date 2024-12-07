import math
import logging
import torch
import numpy as np
from fitxf.math.fit.utils.FitUtils import FitUtils
from fitxf.math.utils.Logging import Logging
from torch.utils.data import DataLoader, RandomSampler


class TestModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.h1 = torch.nn.Linear(in_features=3, out_features=100)
        self.h2 = torch.nn.ReLU()
        self.h3 = torch.nn.Linear(in_features=100, out_features=1)
        return

    def forward(self, X):
        y1 = self.h1(X)
        y2 = self.h2(y1)
        y3 = self.h3(y2)
        return y3


class FitUtilsUt:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def _test_pred_helper(
            self,
            test_name,
            X_test,
            Y_test,
            # list of tuples (4 elements: normalize_or_not, cluster_centers, predicted_classes, predicted_distances)
            test_set,
            mean_per_label,
            remove_dup_lbl,
    ):
        # if not test_name in ['Test 2 Mean Per Label #1']:
        #     return
        fut = FitUtils(logger=self.logger)
        for (nm, exp_val, exp_cls, exp_dist) in test_set:
            mean_ts = fut.fit_mean(
                X=X_test, labels=Y_test, mean_per_label=mean_per_label, normalize_X=nm
            )
            mean_ts_list = torch.round(mean_ts, decimals=5).tolist()
            exp_val_list = torch.round(exp_val, decimals=5).tolist()
            # Test by class
            for class_id in torch.unique(Y_test):
                # index must be integral
                class_id = int(class_id.item())
                # convert to list and sort for consistency, because the clusters may return in non-specific ordering
                mean_tx_class_id = mean_ts_list[class_id]
                mean_tx_class_id.sort()
                exp_val_class_id = exp_val_list[class_id]
                exp_val_class_id.sort()
                print(
                    'Checking class no. ' + str(class_id) + ' Fitted mean per label = ' + str(mean_per_label)
                    +  ', normalize = ' + str(nm) + ' with X:\n' + str(X_test[Y_test==class_id])
                    + '\nresult fit:\n' + str(mean_tx_class_id) + '\nexpected:\n' + str(exp_val_class_id)
                )
                # element wise difference, all must be exactly 0.
                test_tsr = torch.round(torch.Tensor(mean_tx_class_id), decimals=4) - torch.Tensor(exp_val_class_id)
                test_res = torch.sum(test_tsr != 0.) == 0
                assert test_res, \
                    ('<' + str(test_name) + '> For class no. ' + str(class_id) + ', normalize=' + str(nm)
                     + ', expect mean ' + str(exp_val_class_id) + ' observed ' + str(mean_tx_class_id))

            test_top_k = 3
            c, d = fut.predict_mean(
                X              = X_test,
                mean_tensors   = mean_ts,
                reps_per_label = mean_per_label,
                remove_dup_lbl = remove_dup_lbl,
                normalize_X    = nm,
            )
            # print('***** Predicted classes')
            # print(c)
            if exp_cls is not None:
                for i, c_row in enumerate(c):
                    # Test top k classes will do
                    if i >= test_top_k:
                        break
                    test_tsr = torch.sum((c_row - exp_cls[i]) != 0.)
                    test_res = test_tsr == 0.
                    assert test_res, \
                        '<' + str(test_name) + '> Row #' + str(i) + ' For normalize=' \
                        + str(nm) + ', test tensor:\n' + str(test_tsr) \
                        + '\n, expect classes:\n' + str(exp_cls[i]) + '\nobserved:\n' + str(c_row)
            if exp_dist is not None:
                for i, d_row in enumerate(d):
                    # Test top k classes will do
                    if i >= test_top_k:
                        break
                    test_tsr = torch.sum((torch.round(d_row, decimals=4) - exp_dist[i]) != 0.)
                    test_res = test_tsr == 0.
                    assert test_res, \
                        '<' + str(test_name) + '> Row #' + str(i) + ' For normalize=' + str(nm) \
                        + ', test tensor:\n' + str(test_tsr) + '\n, expect dist:\n' + str(exp_dist[i]) \
                        + '\nobserved:\n' + str(d_row)

    def test_predict(
            self,
    ):
        _, _, lbl_mapped = FitUtils().map_labels_to_consecutive_numbers(lbl_list=[4, 2, 6, 6, 6, 55, 0])
        expected_res = [2, 1, 3, 3, 3, 4, 0]
        assert lbl_mapped == expected_res, \
            'Test map labels to consecutive numbers, observed ' + str(lbl_mapped) + ' expected ' + str(expected_res)

        X = torch.Tensor([
            # class 0
            [0, 1], [1, 2], [2, 3], [4, 5], [5, 6],
            # class 1
            [6, 7], [7, 8], [8, 9],
            # class 2
            [9, 10], [10, 11],
        ])
        # at_msk = torch.Tensor([True] * 10)
        Y = torch.Tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
        # class mean tensors after classification
        expected_value = torch.Tensor([
            [2.4000, 3.4000],
            [7.0000, 8.0000],
            [9.5000, 10.5000],
        ])
        # normalized class mean tensors
        expected_value_nm = torch.Tensor([
            [0.4534, 0.8551],
            [0.6579, 0.7531],
            [0.6708, 0.7416],
        ])
        expected_classes = torch.IntTensor([
            [0., 1., 2.],
            [0., 1., 2.],
            [0., 1., 2.],
            [0., 1., 2.],
            # with only 1 mean per label, the 5th point is predicted wrong
            [1., 0., 2.],
            [1., 2., 0.],
            [1., 2., 0.],
            [1., 2., 0.],
            [2., 1., 0.],
            [2., 1., 0.]
        ])
        # normalizing will screw up predictions
        expected_classes_nm = torch.IntTensor([
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [1, 2, 0],
            [1, 2, 0],
            [1, 2, 0],
            [1, 2, 0],
            [2, 1, 0],
            [2, 1, 0],
            [2, 1, 0]
        ])
        # from euclidean distance
        expected_distances = torch.Tensor([
            [3.3941, 9.8995, 13.4350],
            [1.9799, 8.4853, 12.0208],
            [0.5657, 7.0711, 10.6066],
            [2.2627, 4.2426, 7.7782],
            [2.8284, 3.6770, 6.3640],
            [1.4142, 4.9497, 5.0912],
            [0.0000, 3.5355, 6.5054],
            [1.4142, 2.1213, 7.9196],
            [0.7071, 2.8284, 9.3338],
            [0.7071, 4.2426, 10.7480]
        ])
        # from dot product
        expected_distances_nm = torch.Tensor([
            [0.8551, 0.7531, 0.7416],
            [0.9678, 0.9676, 0.9633],
            [0.9915, 0.9892, 0.9630],
            [0.9990, 0.9982, 0.9509],
            [0.9997, 0.9992, 0.9471],
            [0.9999, 0.9996, 0.9443],
            [1.0000, 0.9999, 0.9421],
            [1.0000, 0.9999, 0.9403],
            [1.0000, 0.9999, 0.9389],
            [1.0000, 0.9998, 0.9377]
        ])
        self._test_pred_helper(
            test_name = 'Test 1 Mean Per Label',
            X_test   = X,
            Y_test   = Y,
            test_set = [
                (False, expected_value, expected_classes, expected_distances),
                (True, expected_value_nm, expected_classes_nm, expected_distances_nm),
            ],
            mean_per_label = 1,
            remove_dup_lbl = False,
        )

        expected_value2 = torch.Tensor([
            [[4.5000, 5.5000], [1.0000, 2.0000]],
            [[6.5000, 7.5000], [8.0000, 9.0000]],
            [[9., 10.], [10., 11.]],
        ])
        expected_value2_nm = torch.Tensor([
            [[0.5667, 0.8189], [0.0000, 1.0000]],
            [[0.6508, 0.7593], [0.6614, 0.7500]],
            [[0.6690, 0.7433], [0.6727, 0.7399]],
        ])
        expected_classes2 = torch.IntTensor([
            [0, 0, 1, 1, 2],
            [0, 0, 1, 1, 2],
            [0, 0, 1, 1, 2],
            [0, 1, 0, 1, 2],
            [0, 1, 1, 0, 2],
            [1, 0, 1, 2, 2],
            [1, 1, 2, 0, 2],
            [1, 2, 1, 2, 0],
            [2, 1, 2, 1, 0],
            [2, 2, 1, 1, 0]
        ])
        expected_classes_rmdup2 = torch.IntTensor([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [1, 2, 0],
            [2, 1, 0],
            [2, 1, 0]
        ])
        self._test_pred_helper(
            test_name = 'Test 2 Mean Per Label #1',
            X_test   = X,
            Y_test   = Y,
            test_set = [
                (False, expected_value2, expected_classes2, None),
                (True, expected_value2_nm, None, None),
            ],
            mean_per_label = 2,
            remove_dup_lbl = False,
        )
        # with remove duplicate predictions
        self._test_pred_helper(
            test_name = 'Test 2 Mean Per Label #2',
            X_test   = X,
            Y_test   = Y,
            test_set = [
                (False, expected_value2, expected_classes_rmdup2, None),
                (True, expected_value2_nm, None, None),
            ],
            mean_per_label = 2,
            remove_dup_lbl = True,
        )

        #
        # Test for cluster centers more than points
        #
        expected_value3 = torch.Tensor([
            [[0.5000, 1.5000], [4.5000, 5.5000], [2.0000, 3.0000]],
            [[6.0000, 7.0000], [8.0000, 9.0000], [7.0000, 8.0000]],
            [[9.0, 10.0],      [9.0, 10.0],      [10.0, 11.0]],
        ])
        expected_value3_nm = torch.Tensor([
            [[0.6065, 0.7937], [0.0000, 1.0000], [0.4472, 0.8944]],
            [[0.6508, 0.7593], [0.6644, 0.7474], [0.6585, 0.7526]],
            [[0.6690, 0.7433], [0.6690, 0.7433], [0.6727, 0.7399]],
        ])
        # when mean tensors not normalized & prediction using euclid-dist, all predictions correct
        expected_classes3 = torch.IntTensor([
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 2],
            [1, 1, 1, 2, 0],
            [1, 2, 1, 1, 2],
            [2, 2, 2, 1, 1],
            [2, 2, 2, 1, 1]
        ])
        expected_classes_rmdup3 = torch.IntTensor([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [1, 2, 0],
            [2, 1, 0],
            [2, 1, 0]
        ])
        expected_classes_rmdup3_nm = torch.IntTensor([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [1, 0, 2],    # wrong pred
            [1, 2, 0],    # wrong pred
            [1, 2, 0],
            [1, 2, 0],
            [1, 2, 0],
            [2, 1, 0],
            [2, 1, 0]
        ])
        expected_classes3_nm = torch.IntTensor([
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 0, 1, 1],
            [1, 1, 0, 1, 2],    # wrong pred
            [1, 1, 1, 2, 2],    # wrong pred
            [1, 1, 1, 2, 2],
            [1, 1, 1, 2, 2],
            [1, 2, 1, 2, 2],
            [2, 2, 2, 1, 1],
            [2, 2, 2, 1, 1]
        ])
        expected_distances3 = torch.Tensor([
            [0.7071, 2.8284, 6.3640, 8.4853, 9.8995],
            [0.7071, 1.4142, 4.9497, 7.0711, 8.4853],
            [0.0000, 2.1213, 3.5355, 5.6569, 7.0711],
            [0.7071, 2.8284, 2.8284, 4.2426, 4.9497],
            [0.7071, 1.4142, 2.8284, 4.2426, 4.2426],
            [0.0000, 1.4142, 2.1213, 2.8284, 4.2426],
            [0.0000, 1.4142, 1.4142, 2.8284, 3.5355],
            [0.0000, 1.4142, 1.4142, 2.8284, 2.8284],
            [0.0000, 1.4142, 1.4142, 1.4142, 2.8284],
            [0.0000, 0.0000, 1.4142, 2.8284, 4.2426]
        ])
        expected_distances_rmdup3 = torch.Tensor([
            [0.7071, 8.4853, 12.7279],
            [0.7071, 7.0711, 11.3137],
            [0.0000, 5.6569, 9.8995],
            [0.7071, 2.8284, 7.0711],
            [0.7071, 1.4142, 5.6569],
            [0.0000, 2.1213, 4.2426],
            [0.0000, 2.8284, 3.5355],
            [0.0000, 1.4142, 4.9497],
            [0.0000, 1.4142, 6.3640],
            [0.0000, 2.8284, 7.7782]
        ])
        # from dot product
        expected_distances3_nm = torch.Tensor([
            [1.0000, 0.8944, 0.7937, 0.7593, 0.7526],
            [1.0000, 0.9812, 0.9701, 0.9676, 0.9656],
            [0.9968, 0.9927, 0.9923, 0.9915, 0.9904],
            [0.9994, 0.9990, 0.9987, 0.9987, 0.9983],
            [0.9999, 0.9997, 0.9995, 0.9993, 0.9991],
            [1.0000, 0.9999, 0.9998, 0.9997, 0.9996],
            [1.0000, 1.0000, 0.9999, 0.9999, 0.9998],
            [1.0000, 1.0000, 1.0000, 0.9999, 0.9999],
            [1.0000, 1.0000, 1.0000, 1.0000, 0.9999],
            [1.0000, 1.0000, 1.0000, 0.9999, 0.9998]
        ])
        expected_distances_rmdup3_nm = torch.Tensor([
            [1.0000, 0.7593, 0.7433],
            [1.0000, 0.9701, 0.9640],
            [0.9968, 0.9927, 0.9895],
            [0.9994, 0.9987, 0.9983],
            [0.9999, 0.9993, 0.9980],
            [1.0000, 0.9997, 0.9974],
            [1.0000, 0.9999, 0.9967],
            [1.0000, 1.0000, 0.9962],
            [1.0000, 1.0000, 0.9957],
            [1.0000, 0.9999, 0.9953]
        ])
        self._test_pred_helper(
            test_name = 'Test 3 Mean Per Label (remove dup lbl = False)',
            X_test   = X,
            Y_test   = Y,
            test_set = [
                (False, expected_value3, expected_classes3, expected_distances3),
                (True, expected_value3_nm, expected_classes3_nm, expected_distances3_nm),
            ],
            mean_per_label = 3,
            remove_dup_lbl = False,
        )
        self._test_pred_helper(
            test_name = 'Test 3 Mean Per Label (remove dup lbl = True)',
            X_test   = X,
            Y_test   = Y,
            test_set = [
                (False, expected_value3, expected_classes_rmdup3, expected_distances_rmdup3),
                (True, expected_value3_nm, expected_classes_rmdup3_nm, expected_distances_rmdup3_nm),
            ],
            mean_per_label = 3,
            remove_dup_lbl = True,
        )

    def test_nn(self, epochs=5000, plot_graph=False, tol_dif=0.1):
        fu = FitUtils(logger=self.logger)
        """
        Sample train, train to recognize binary representation
        """
        X = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ], dtype=float)
        Y = np.array([
            [0], [1], [2], [3], [4], [5], [6], [7],
        ], dtype=float)
        # attn_masks = np.array([
        #     [1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1],
        # ], dtype=float)

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        # attn_masks = torch.from_numpy(attn_masks).float()

        train_set, val_set = fu.training_data_split(
            val_ratio  = 0.0,
            inputs     = X,
            labels     = Y,
            attn_masks = None,
            return_tensors = 'pt',
        )

        # Prepare DataLoader
        # Total step/batch should be int(n_train_samples/batch_size)
        dl = DataLoader(
            train_set,
            sampler = RandomSampler(train_set),
            batch_size = len(X),
        )

        for acc_steps in (1, len(X)/2,):
            acc_steps = math.floor(acc_steps)

            model = TestModel()
            model.train(mode=True)

            losses = fu.torch_train(
                train_dataloader = dl,
                model = model,
                loss_func = torch.nn.MSELoss(),
                optimizer = torch.optim.SGD(model.parameters(), lr=0.005),
                regularization_type = None,
                epochs = epochs,
                accum_steps = acc_steps,
                plot_losses = plot_graph,
            )
            # Final loss must not exceed difference of 0.01 per data
            tol = 7*(0.1**2)
            assert losses[-1] < tol, 'Final loss ' + str(losses[-1]) + ' must be less than ' + str(tol)
            print(
                'Accum steps ' + str(acc_steps) + ', initial loss ' + str(losses[0])
                + ' to final loss ' + str(losses[-1]) + ', tolerance ' + str(tol)
            )

            model.eval()
            pred = model(X)
            print('Expected/Predicted ' + str(list(zip(Y.tolist(), torch.round(pred, decimals=3).tolist()))))
            for i in range(len(Y)):
                x = X[i]
                val = Y[i][0]
                val_pred = pred[i][0]
                msg = 'For accum steps ' + str(acc_steps) + ', x=' + str(x) + ', expected ' + str(val)\
                      + ', got ' + str(val_pred) + ', tol=' + str(tol_dif)
                print(msg)
                assert abs(val_pred - val) < tol_dif, msg

    def test_dist(self):
        fu = FitUtils(logger=self.logger)
        x = np.array([
            [0., 0.], [0., 1.], [1., 0.],
        ])
        x_ref = np.array([0.2, 0.5])
        for ref, method, metric, exp_mean, exp_dists in [
            (None,  'exact', 'euclid',        1.138071187, np.array([1., 1., 1.41421356])),
            (None,  'exact', 'euclid-square', 4/3,         np.array([1., 1., 2.])),
            (x_ref, 'exact', 'euclid',        0.673477024, np.array([0.53851648, 0.53851648, 0.94339811])),
            (x_ref, 'exact', 'euclid-square', 0.49,         np.array([0.29, 0.29, 0.89])),
            (None,  'mc',    'euclid',        1.138071187, None),
            (None,  'mc',    'euclid-square', 4/3,         None),
            (x_ref, 'mc',    'euclid',        0.673477024, None),
            (x_ref, 'mc',    'euclid-square', 0.49,        None),
        ]:
            dists = fu.get_point_distances(
                np_tensors = x,
                np_center  = ref,
                method     = method,
                metric     = metric,
                mc_iters   = 10000,
            )
            dists_mean = np.mean(dists)
            print('Method', method, 'metric', metric, dists[0:min(30,len(dists))])
            print('Mean', dists_mean)
            if exp_dists is not None:
                assert np.sum(np.round(exp_dists - dists, 4)**2) == 0., \
                    'Distances method "' + str(method) + '", metric "' + str(metric) + '":' + str(dists)
            assert round((exp_mean - dists_mean)**2, 2) == 0., \
                'Mean distance method "' + str(method) + '", metric "' + str(metric) + '":' + str(dists_mean)

    def test_map_to_connbr(self):
        fu = FitUtils(logger=self.logger)
        labels = ['drama', 'comedy', 'action', 'comedy', 'action', 'drama']
        test_sets = [
            {
                'curmap': None,
                'expected_map': {'action': 0, 'comedy': 1, 'drama': 2},
            },
            {
                'curmap': {'comedy': 1},
                'expected_map': {'comedy': 1, 'action': 0, 'drama': 2},
            },
            {
                'curmap': {'comedy': 99},
                'expected_map': {'comedy': 99, 'action': 0, 'drama': 1},
            },
            {
                'curmap': {'horror': 99},
                'expected_map': {'action': 0, 'comedy': 1, 'drama': 2, 'horror': 99},
            },
            {
                'curmap': {'action': 77, 'comedy': 88, 'drama': 99},
                'expected_map': {'action': 77, 'comedy': 88, 'drama': 99},
            },
            {
                'curmap': {'comedy': 88, 'drama': 99, 'horror': 111},
                'expected_map': {'action': 0, 'comedy': 88, 'drama': 99, 'horror': 111},
            },
            {
                # Should throw exception with duplicate values
                'curmap': {'comedy': 0, 'drama': 0, 'horror': 1},
                'expected_map': None,
            },
        ]
        for i, s in enumerate(test_sets):
            curmap, exp_map_lbl = s['curmap'], s['expected_map']
            try:
                map_lbl = fu.map_labels_to_numbers(lbl_list=labels, existing_label_map=curmap, strict_1_to_1=True)
            except:
                map_lbl = None
            assert map_lbl == exp_map_lbl, \
                '# ' + str(i) + ' Observed map ' + str(map_lbl) + ' different from expected ' + str(exp_map_lbl)

    def test(self):
        self.test_map_to_connbr()
        self.test_nn(epochs=5000, plot_graph=False, tol_dif=0.1)
        self.test_predict()
        self.test_dist()
        print('ALL TESTS PASSED')


if __name__ == '__main__':
    fu = FitUtilsUt(logger=Logging.get_default_logger(log_level=logging.INFO, propagate=False))
    fu.test()
    exit(0)
