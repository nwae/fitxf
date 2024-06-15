import logging
import os
from io import StringIO
import numpy as np
import pandas as pd
from fitxf.math.algo.encoding.Base64 import Base64
from fitxf.math.data.ut.LabelTextEmbed01 import DATA_LABEL_TEXT_EMBEDDING_01_TRAIN, DATA_LABEL_TEXT_EMBEDDING_01_EVAL
from fitxf.math.fit.transform.FitXformInterface import FitXformInterface
from fitxf.math.fit.transform.FitXformPca import FitXformPca
from fitxf.math.fit.transform.FitXformCluster import FitXformCluster, FitXformClusterCosine
from fitxf.math.lang.encode.LangModelPt import LangModelPt as LmPt
from fitxf.math.utils.EnvironRepo import EnvRepo
from fitxf.math.utils.Logging import Logging


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
        for F, score_thr, ret_full_rec in [
            (FitXformPca, 0.95, False),
            (FitXformPca, 0.95, True),
            (FitXformCluster, 0.95, False),
            (FitXformClusterCosine, 0.95, False),
        ]:
            self.__test_fit(
                fitter_name = str(F.__class__),
                FitterClassType = F,
                avg_score_threshold = score_thr,
                ret_full_rec = ret_full_rec,
            )
        return

    def __test_fit(
            self,
            fitter_name: str,
            FitterClassType: type(FitXformInterface),
            avg_score_threshold: float,
            ret_full_rec: bool,
    ):
        fitter = FitterClassType(logger=self.logger)
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
        full_recs_train = [
            {k: v for (k, v) in r.items() if k not in ['embedding']} for r in df_train.to_dict(orient='records')
        ]

        def get_lm() -> LmPt:
            return LmPt.get_singleton(
                LmClass = LmPt,
                lang = 'en',
                cache_folder = self.lm_cache_folder,
                logger = self.logger,
            )

        try:
            raise Exception('Force to use pre-calculated embeddings.')
            # emb_train = get_lm().encode(text_list=texts_train, return_tensors='np')
            # emb_eval = get_lm().encode(text_list=texts_test, return_tensors='np')
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
            X_full_records = full_recs_train,
            target_grid_density = 2.,
            measure = 'min',
            min_components = 3,
            # estimate n centers at most 2x number of unique labels
            max_components = len(np.unique(labels_train)) * 2,
        )

        self.__test_predictions(
            fitter_name = fitter_name,
            fitter = fitter,
            emb_train = emb_train,
            emb_eval = emb_eval,
            ret_full_rec = ret_full_rec,
            avg_score_threshold = avg_score_threshold,
            expected_top_labels = labels_eval,
        )
        # now load a new instance by saving json and reloading
        fitter_new_loaded = self.__test_save_load_model(
            fitter_old = fitter,
            FitterClassType = FitterClassType,
        )
        # do the same test on this new object
        self.__test_predictions(
            fitter_name = fitter_name,
            fitter = fitter_new_loaded,
            emb_train = emb_train,
            emb_eval = emb_eval,
            # we no longer have full records
            ret_full_rec = False,
            avg_score_threshold = avg_score_threshold,
            expected_top_labels = labels_eval,
        )
        print('ALL TESTS PASSED OK')
        return

    def __test_predictions(
            self,
            fitter_name: str,
            fitter: FitXformInterface,
            emb_train: np.ndarray,
            emb_eval: np.ndarray,
            ret_full_rec: bool,
            avg_score_threshold: float,
            expected_top_labels: list,
    ):
        x_transform = fitter.X_transform
        x_transform_check = fitter.transform(X=emb_train)
        x_inverse_transform = fitter.inverse_transform(X=x_transform)

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

            print(
                '[' + str(fitter_name) + '] Use grid "' + str(use_grid) + '", predicted labels: ' + str(pred_labels)
            )
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
                '[' + str(fitter.__class__) + '] Use grid "' + str(use_grid) + '". Mean score '
                + str(score_avg) + ', scores' + str(scores)
            )
            assert score_avg > avg_score_threshold, \
                '[' + str(fitter.__class__) + '] Use grid "' + str(use_grid) + '". Mean score fail ' + str(score_avg) \
                + ' < ' + str(avg_score_threshold) + '. Scores ' + str(scores)

    def __test_save_load_model(
            self,
            fitter_old: FitXformInterface,
            FitterClassType: type(FitXformInterface),
    ):
        #
        # Test model saving to json and reloading is ok
        #
        centers_before = np.array(fitter_old.model_centers)
        pca_before = np.array(fitter_old.model_principal_components)
        model_dict = fitter_old.model_to_json(numpy_to_base64_str=True)
        [self.logger.info(str(k) + ': ' + str(v)) for k, v in model_dict.items()]

        def get_fitter() -> FitXformInterface:
            return FitterClassType(logger=self.logger)

        fitter_new = get_fitter()
        fitter_new.load_model_from_json(model_json=model_dict)

        assert fitter_new.model_centers.shape == centers_before.shape, \
            'Shape before ' + str(centers_before.shape) + ' but after ' + str(fitter_new.model_centers.shape)
        assert fitter_new.model_principal_components.shape == pca_before.shape, \
            'Shape before ' + str(pca_before.shape) + ' but after ' + str(fitter_new.model_principal_components.shape)

        diff = np.sum( (fitter_new.model_centers - centers_before) ** 2 )
        assert diff < 0.0000000001, \
                'Centers after and before different, before\n' + str(centers_before) \
                + ', after\n' + str(fitter_new.model_centers)
        diff = np.sum( (fitter_new.model_principal_components - pca_before) ** 2 )
        assert diff < 0.0000000001, \
                'Principle components after and before different, before\n' + str(pca_before) \
                + ', after\n' + str(fitter_new.model_principal_components)
        return fitter_new


if __name__ == '__main__':
    FitXformUnitTest(
        lm_cache_folder = EnvRepo(repo_dir=os.environ.get("REPO_DIR", None)).MODELS_PRETRAINED_DIR,
        logger = Logging.get_default_logger(log_level=logging.INFO, propagate=False),
    ).test()
    exit(0)
