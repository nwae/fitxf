import logging
import os
from io import StringIO
import numpy as np
import pandas as pd
from nwae.math.algo.encoding.Base64 import Base64
from nwae.math.data.ut.LabelTextEmbed01 import DATA_LABEL_TEXT_EMBEDDING_01_TRAIN, DATA_LABEL_TEXT_EMBEDDING_01_EVAL
from nwae.math.fit.transform.FitXformInterface import FitXformInterface
from nwae.math.fit.transform.FitXformPca import FitXformPca
from nwae.math.fit.transform.FitXformCluster import FitXformCluster, FitXformClusterCosine
from nwae.math.lang.encode.LangModelPt import LangModelPt as LmPt
from nwae.math.utils.EnvironRepo import EnvRepo
from nwae.math.utils.Logging import Logging


class FitXformUnitTest:

    def __init__(
            self,
            lm_cache_folder = None,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.lm_cache_folder = lm_cache_folder
        self.base64 = Base64(logger=self.logger)
        return

    def test(self):
        for f, score_thr, ret_full_rec in [
            (FitXformPca(logger=self.logger), 0.9, False),
            (FitXformPca(logger=self.logger), 0.9, True),
            (FitXformCluster(logger=self.logger), 0.5, False),
            (FitXformClusterCosine(logger=self.logger), 0.59, False),
        ]:
            self.__test_fit(
                fitter_name = str(f.__class__),
                fitter = f,
                avg_score_threshold = score_thr,
                ret_full_rec = ret_full_rec,
            )
        return

    def __test_fit(
            self,
            fitter_name: str,
            fitter: FitXformInterface,
            avg_score_threshold: float,
            ret_full_rec: bool,
    ):
        # texts_train = [
        #     "Let's have coffee", "Free for a drink?", "How about Starbucks?",
        #     "I am busy", "Go away", "Don't disturb me",
        #     "Monetary policies", "Interest rates", "Deposit rates",
        # ]
        # labels_train = ['drink', 'drink', 'drink', 'busy', 'busy', 'busy', 'finance', 'finance', 'finance']
        # texts_test = ["How about a cup of coffee?", "Not now", "Financial policies"]

        def get_data(
                s,
        ):
            df = pd.read_csv(
                filepath_or_buffer = StringIO(s),
                sep = ',',
                index_col = False,
            )
            columns_keep = ['label', 'text', 'embedding']
            for c in columns_keep: assert c in df.columns
            df = df[columns_keep]
            df.dropna(inplace=True)
            # _, _, df[self.col_label_std] = FitUtils().map_labels_to_consecutive_numbers(lbl_list=list(df[self.col_label]))
            self.logger.info('Successfully read data of shape ' + str(df.shape))
            return df

        df_train = get_data(s=DATA_LABEL_TEXT_EMBEDDING_01_TRAIN)
        df_eval = get_data(s=DATA_LABEL_TEXT_EMBEDDING_01_EVAL)

        texts_train, labels_train = df_train['text'].tolist(), df_train['label'].tolist()
        texts_eval, labels_eval = df_eval['text'].tolist(), df_eval['label'].tolist()

        def get_lm() -> LmPt:
            return LmPt.get_singleton(
                LmClass = LmPt,
                lang = 'en',
                cache_folder = self.lm_cache_folder,
                logger = self.logger,
            )

        try:
            raise Exception('Force to use pre-calculated embeddings.')
            emb_train = get_lm().encode(text_list=texts_train, return_tensors='np')
            emb_eval = get_lm().encode(text_list=texts_test, return_tensors='np')
        except Exception as ex:
            self.logger.info('Failed to calculate embeddings: ' + str(ex) + ', using precalculated embeddings instead.')
            emb_train = np.array([
                self.base64.decode_base64_string_to_numpy_array(s64=s, data_type=np.float64)
                for s in df_train['embedding'].tolist()
            ])
            emb_eval = np.array([
                self.base64.decode_base64_string_to_numpy_array(s64=s, data_type=np.float64)
                for s in df_eval['embedding'].tolist()
            ])
        # x = np.array([
        #     [1, 1, 1, 1], [2, 2, 2, 2],
        #     [2, 1.5, -1, 0.3], [1, 2, -2, 1],
        #     [3, 0.5, 0, -1], [1, 1, 1, -2],
        # ])
        res = fitter.fit_optimal(
            X = emb_train,
            X_labels = labels_train,
            X_full_records = [
                {k: v for (k, v) in r.items() if k not in ['embedding']} for r in df_train.to_dict(orient='records')
            ],
            target_grid_density = 2.,
            measure = 'min',
            min_components = 3,
            max_components = 3,
        )
        self.logger.debug('Fitter "' + str(fitter_name) + '" fit result ' + str(res))
        print('grid numbers', fitter.X_grid_numbers)
        print('distance error', fitter.distance_error)
        print('distance error mean', fitter.distance_error_mean)
        print('angle error', fitter.angle_error)
        print('angle error mean', fitter.angle_error_mean)
        print('grid density', fitter.grid_density)
        print('grid density mean', fitter.grid_density_mean)

        x_transform = fitter.X_transform
        x_transform_check = fitter.X_transform_check
        x_inverse_transform = fitter.X_inverse_transform

        # Check if estimation of actual value is correct
        diff = x_inverse_transform - emb_train
        sq_err_per_vect = np.sum(diff*diff) / len(diff)
        x_dim = emb_train.shape[-1]
        sq_err_per_vect_thr = 0.1 * x_dim
        assert sq_err_per_vect < sq_err_per_vect_thr, \
            '[' + str(fitter_name) + '] Estimate back using PCA, per vector sq err ' + str(sq_err_per_vect) \
            + '>=' + str(sq_err_per_vect_thr) + ', details: ' + str(diff)

        # Check if our manual calculation of the principal component values are correct
        diff = x_transform_check - x_transform
        sq_err_sum = np.sum(diff*diff)
        assert sq_err_sum < 0.000000001, \
            '[' + str(fitter_name) + '] Manual calculate PCA component values, sq err ' + str(sq_err_sum) \
            + ', diff ' + str(diff)

        for use_grid in (False, True,):
            pred_labels, pred_probs = fitter.predict(
                X = emb_eval,
                use_grid = use_grid,
                return_full_record = ret_full_rec,
                top_k = 2,
            )
            # pred_labels_full, pred_probs_full = fitter.predict(
            #     X = emb_eval,
            #     use_grid = use_grid,
            #     return_full_record = True,
            #     top_k = 2,
            # )
            # print(pred_labels, pred_probs)
            # print([[r['label'] for r in row_recs] for row_recs in pred_labels_full], pred_probs_full)
            # assert pred_labels ==[[r['label'] for r in row_recs] for row_recs in pred_labels_full], \
            #     '[' + str(fitter_name) + '] Use grid "' + str(use_grid) + \
            #     '". Predict without full records must be same with prediction with full records:\n' \
            #     + str(pred_labels) + '\n' + str([[r['label'] for r in row_recs] for row_recs in pred_labels_full])
            # assert pred_probs == pred_probs_full, \
            #     '[' + str(fitter_name) + '] Use grid "' + str(use_grid) + '"' + 'Probs also must equal.'
            #
            # if ret_full_rec:
            #     pred_labels = pred_labels_full
            #     pred_probs = pred_probs_full

            print(
                '[' + str(fitter_name) + '] Use grid "' + str(use_grid) + '", predicted labels: ' + str(pred_labels)
            )
            expected_top_labels = df_eval['label'].tolist()
            scores = []
            for i, exp_lbl in enumerate(expected_top_labels):
                pred_top_label = pred_labels[i][0]['label'] if ret_full_rec else pred_labels[i][0]
                pred_top_label_2 = pred_labels[i][1]['label'] if ret_full_rec else pred_labels[i][1]
                if type(fitter) in [FitXformClusterCosine, FitXformCluster]:
                    # First in tuple is predicted cluster number, take
                    pred_top_label = pred_top_label['user_label_estimate']
                    pred_top_label_2 = pred_top_label_2['user_label_estimate']
                # 1.0 for being 1st, 0.5 for appearing 2nd
                score_i = 1*(pred_top_label == exp_lbl) + 0.5*(pred_top_label_2 == exp_lbl)
                score_i = min(score_i, 1.0)
                scores.append(score_i)
                # Check only top prediction
                if pred_top_label != exp_lbl:
                    self.logger.warning(
                        '[' + str(fitter_name) + '] #' + str(i) + ' Use grid "' + str(use_grid)
                        + '". Predicted top label "' + str(pred_top_label) + '" not expected "' + str(exp_lbl) + '"'
                    )
            score_avg = np.mean(np.array(scores))
            self.logger.info(
                '[' + str(fitter_name) + '] Use grid "' + str(use_grid) + '". Mean score '
                + str(score_avg) + ', scores' + str(scores)
            )
            assert score_avg > avg_score_threshold, \
                '[' + str(fitter_name) + '] Use grid "' + str(use_grid) + '". Mean score fail ' + str(score_avg) \
                + ' < ' + str(avg_score_threshold) + '. Scores ' + str(scores)

        print('ALL TESTS PASSED OK')
        return


if __name__ == '__main__':
    FitXformUnitTest(
        lm_cache_folder = EnvRepo(repo_dir=os.environ.get("REPO_DIR", None)).MODELS_PRETRAINED_DIR,
        logger = Logging.get_default_logger(log_level=logging.INFO, propagate=False),
    ).test()
    exit(0)
