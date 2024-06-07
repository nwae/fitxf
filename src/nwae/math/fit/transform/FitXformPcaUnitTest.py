import logging
import numpy as np
from nwae.math.fit.transform.FitXformPca import FitXformPca
from nwae.math.lang.encode.LangModelPt import LangModelPt as LmPt
from nwae.math.utils.EnvironRepo import EnvRepo
from nwae.math.utils.Logging import Logging


class FitXformPcaUnitTest:

    def __init__(
            self,
            lm_cache_folder = None,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.lm_cache_folder = lm_cache_folder
        return

    def test(self):
        fitter = FitXformPca(logger=self.logger)

        texts_train = [
            "Let's have coffee", "Free for a drink?", "How about Starbucks?",
            "I am busy", "Go away", "Don't disturb me",
            "Monetary policies", "Interest rates", "Deposit rates",
        ]
        labels_train = ['drink', 'drink', 'drink', 'busy', 'busy', 'busy', 'finance', 'finance', 'finance']
        texts_test = ["How about a cup of coffee?", "Not now", "Financial policies"]

        lmo = LmPt(
            lang = 'en',
            cache_folder = self.lm_cache_folder,
            logger = self.logger,
        )

        x = lmo.encode(text_list=texts_train, return_tensors='np')
        x_test = lmo.encode(text_list=texts_test, return_tensors='np')
        # x = np.array([
        #     [1, 1, 1, 1], [2, 2, 2, 2],
        #     [2, 1.5, -1, 0.3], [1, 2, -2, 1],
        #     [3, 0.5, 0, -1], [1, 1, 1, -2],
        # ])
        res = fitter.fit_optimal(
            X = x,
            X_labels = labels_train,
            target_grid_density = 2.,
            measure = 'min',
            min_components = 3,
            max_components = 3,
        )
        self.logger.debug('PCA fit result ' + str(res))
        print('grid numbers', fitter.X_grid_numbers)
        print('distance error', fitter.distance_error)
        print('distance error %', fitter.distance_error_pct)
        print('distance error mean', fitter.distance_error_mean)
        print('angle error', fitter.angle_error)
        print('angle error mean', fitter.angle_error_mean)
        print('grid density', fitter.grid_density)
        print('grid density mean', fitter.grid_density_mean)

        x_pca = fitter.X_transform
        x_pca_check = fitter.X_transform_check
        x_estimate = fitter.X_inverse_transform

        # Check if estimation of actual value is correct
        diff = x_estimate - x
        sq_err_per_vect = np.sum(diff*diff) / len(diff)
        x_dim = x.shape[-1]
        sq_err_per_vect_thr = 0.1 * x_dim
        assert sq_err_per_vect < sq_err_per_vect_thr, \
            'Estimate back using PCA, per vector sq err ' + str(sq_err_per_vect) \
            + '>=' + str(sq_err_per_vect_thr) + ', details: ' + str(diff)

        # Check if our manual calculation of the principal component values are correct
        diff = x_pca_check - x_pca
        sq_err_sum = np.sum(diff*diff)
        assert sq_err_sum < 0.000000001, \
            'Manual calculate PCA component values, sq err ' + str(sq_err_sum) + ', diff ' + str(diff)

        for use_grid in (False, True,):
            pred_labels, pred_probs = fitter.predict(
                X = x_test,
                use_grid = use_grid,
                return_full_record = False,
            )
            print(
                'Use grid "' + str(use_grid) + '", predicted labels: ' + str(pred_labels)
            )
            expected_top_labels = ['drink', 'busy', 'finance']
            for i, exp_lbl in enumerate(expected_top_labels):
                # Check only top prediction
                assert pred_labels[i][0] == exp_lbl, \
                    '#' + str(i) + ' Use grid "' + str(use_grid) + '". Predicted top label ' + str(pred_labels[i]) \
                    + ' not expected ' + str(exp_lbl)

        print('ALL TESTS PASSED OK')
        return


if __name__ == '__main__':
    FitPcaUnitTest(
        lm_cache_folder = EnvRepo(repo_dir=os.environ["REPO_DIR"]).MODELS_PRETRAINED_DIR,
        logger = Logging.get_default_logger(log_level=logging.INFO, propagate=False),
    ).test()
    exit(0)
