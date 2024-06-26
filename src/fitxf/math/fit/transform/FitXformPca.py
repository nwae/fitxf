import logging
import math
import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from fitxf.math.fit.transform.FitXformInterface import FitXformInterface
from fitxf.math.utils.Lock import Lock
from fitxf.math.utils.Env import Env
from fitxf.math.utils.Logging import Logging


#
# Simple and easy to understand article on PCA
# https://ko.wikipedia.org/wiki/%EC%A3%BC%EC%84%B1%EB%B6%84_%EB%B6%84%EC%84%9D
#
# We build these on top of the PCA
#    - Divide the points using PCA components (positive or negative direction) into grids. k components form 2^k grids
#    - Measure grid density (how many points per grid)
#    - Measure distance error of estimated X from PCA & actual X
#    - Use both measures to find optimal number of k, based on measure value specified by user
#
class FitXformPca(FitXformInterface):

    def __init__(
            self,
            logger = None,
    ):
        super().__init__(
            logger = logger,
        )

        self.__mutex_model = 'model'
        self.__lock = Lock(
            mutex_names = [self.__mutex_model],
            logger = self.logger,
        )

        # Model parameters
        self.model_params_ready = False
        return

    def is_model_ready(self):
        return self.model_params_ready

    def __check_consistency(
            self,
            principal_components: np.ndarray,
    ):
        assert self.__lock.is_locked(mutex=self.__mutex_model)

        # Principal components must be of unit length
        tmp = principal_components * principal_components
        pc_lengths = np.sum(tmp, axis=-1) ** 0.5
        expected = np.ones(len(principal_components))
        diff = expected - pc_lengths
        diff = np.sum(diff * diff)
        assert diff < 0.0000000001, 'Expect principal components to be length 1 but got ' + str(pc_lengths)
        return

    def fit_optimal(
            self,
            X: np.ndarray,
            X_labels: list = None,
            X_full_records: list = None,
            target_grid_density = 2,
            # allowed values 'median', 'mean', 'min'
            measure = 'median',
            # Model dependent interpretation, or ignore if not relevant for specific model
            min_components = 1,
            max_components = 99999,
            # pass through mode
            test_mode = False,
    ) -> dict:
        try:
            self.__lock.acquire_mutexes(
                id = 'fit_optimal',
                mutexes = [self.__mutex_model],
            )
            return self.__fit_optimal(
                X = X,
                X_labels = X_labels,
                X_full_records = X_full_records,
                target_grid_density = target_grid_density,
                measure = measure,
                min_components = min_components,
                max_components = max_components,
            )
        finally:
            self.__lock.release_mutexes(mutexes=[self.__mutex_model])

    # Will dynamically look for optimal number of pca components, on the condition of the target grid density
    def __fit_optimal(
            self,
            X: np.ndarray,
            X_labels: list = None,
            X_full_records: list = None,
            target_grid_density = 2,
            # allowed values 'median', 'mean', 'min'
            measure = 'median',
            # Model dependent interpretation, or ignore if not relevant for specific model
            min_components = 1,
            max_components = 99999,
            # pass through mode
            test_mode = False,
    ) -> dict:
        assert target_grid_density > 0, 'Target grid density not valid ' + str(target_grid_density)
        x_dim = X.shape[-1]
        n_high = min(int(x_dim - 1), len(X)- 1)
        n_high = min(max_components, n_high)
        n_high_condition_satisfied = False
        n_low = max(1, min_components)
        n = n_low + math.ceil((n_high - n_low) / 2)

        pca_fit_optimal = None
        while True:
            self.logger.info('Fit PCA with n pca components = ' + str(n))
            pca_fit = self.__fit(
                X = X,
                X_labels = X_labels,
                X_full_records = X_full_records,
                n_components = n,
            )
            self.logger.info(
                'Fit PCA n=' + str(n) + ' returned keys ' + str(pca_fit.keys())
            )
            measure_value = self.grid_density_mean
            if measure_value == 'median':
                measure_value = np.median(self.grid_density)
            elif measure_value == 'min':
                measure_value = np.min(self.grid_density)

            info = 'n pca components = ' + str(n) + ', condition grid density "' + str(measure) \
                   + '" ' + str(measure_value) + ' >= ' + str(target_grid_density)\
                   + ', with distance mean error ' + str(self.distance_error_mean)

            if measure_value >= target_grid_density:
                n_low = n
                # Raise up number of components
                n += math.ceil((n_high - n) / 2)
                self.logger.info(
                    'SATISFIED ' + info + '. low=' + str(n_low) + ', high=' + str(n_high) + ', next n=' + str(n)
                )
                if n >= n_high:
                    # Only move back up if at n=n_high, condition is satisfied. Otherwise we have found our optimum
                    if (n_low < n_high) and n_high_condition_satisfied:
                        n_low = n_high
                        n = n_high
                    else:
                        self.logger.info(
                            'DONE at ' + info + '. low=' + str(n_low) + ', high=' + str(n_high)
                            + ', final grid density "' + str(measure) + '"=' + str(measure_value)
                            # + ', details: ' + str(self.grid_density)
                        )
                        pca_fit_optimal = pca_fit
                        break
            else:
                n_high = n
                n_high_condition_satisfied = False
                # Reduce by half
                n -= math.ceil((n - n_low) / 2)
                self.logger.info(
                    'NOT SATISFIED ' + info + '. low=' + str(n_low) + ', high=' + str(n_high) + ', next n=' + str(n)
                )
                if n <= n_low:
                    # Always take the lower dimension & higher grid density
                    if (n_low < n_high):
                        n_high = n_low
                        n = n_low
                    else:
                        self.logger.info(
                            'Already reached minimum n = ' + str(n_low)
                            + ', optimal not found for target grid density ' + str(target_grid_density)
                            + ', final grid density "' + str(measure) + '"=' + str(measure_value)
                            + ', details: ' + str(self.grid_density)
                        )
                        pca_fit_optimal = pca_fit_optimal
                        break
                continue
        return pca_fit_optimal

    def fit(
            self,
            X: np.ndarray,
            X_labels: list = None,
            X_full_records: list = None,
            # Model dependent interpretation, or ignore if not relevant for specific model
            # For example, can mean how many clusters, or how many PCA components, or how many to sample
            # in a discrete Fourier transform, etc.
            n_components = 2,
            # for fine-tuning already trained clusters, thus no need to start all over again
            # useful for clusters of more than 1,000,000 points for example, where starting
            # again means another half day of fit training
            start_centers: np.ndarray = None,
            # pass through mode
            test_mode = False,
    ) -> dict:
        if X_labels is None:
            X_labels = list(range(len(X)))
        try:
            self.__lock.acquire_mutexes(
                id = 'fit',
                mutexes = [self.__mutex_model],
            )
            return self.__fit(
                X = X,
                X_labels = X_labels,
                X_full_records = X_full_records,
                n_components = n_components,
            )
        finally:
            self.__lock.release_mutexes(mutexes=[self.__mutex_model])

    def __fit(
            self,
            X: np.ndarray,
            X_labels: list = None,
            X_full_records: list = None,
            # Model dependent interpretation, or ignore if not relevant for specific model
            # For example, can mean how many clusters, or how many PCA components, or how many to sample
            # in a discrete Fourier transform, etc.
            n_components = 2,
            # pass through mode
            test_mode = False,
    ):
        assert type(X) is np.ndarray, 'Wrong type X "' + str(type(X)) + '"'
        if X_labels is None:
            X_labels = list(range(len(X)))
        pca = PCA(n_components=n_components)

        # apply principal component analysis to the embeddings table
        # df_reduced = pca.fit_transform(embeddings_df[embeddings_df.columns[:-2]])
        x_reduced = pca.fit_transform(pd.DataFrame(X))

        self.model_params_ready = False

        # Keep model parameters
        self.model_train_total_iterations = None
        self.model_n_components_or_centers = n_components
        # don't keep large tensors in memory
        # self.X = np.array(X)
        self.X_transform = x_reduced
        self.X_labels = X_labels
        self.X_full_records = X_full_records
        self.model_principal_components = pca.components_
        self.model_centroid = np.mean(X, axis=0)
        # not applicable to PCA
        self.model_centers = np.array([])

        self.__check_consistency(principal_components=self.model_principal_components)

        # Estimate back from principal components
        self.X_inverse_transform = self.__inverse_transform(
            x_transform = self.X_transform,
        )
        self.X_grid_vectors, self.X_grid_numbers = self.__calc_pca_grid(
            x_pca = self.X_transform,
        )

        X_lengths = np.sum((X * X), axis=-1) ** 0.5
        X_inverse_lengths = np.sum((self.X_inverse_transform * self.X_inverse_transform), axis=-1) ** 0.5

        # Equivalent to cluster "inertia"
        self.distance_error = np.sum((X - self.X_inverse_transform) ** 2, axis=-1) ** 0.5
        self.distance_error = self.distance_error / X_lengths
        self.distance_error_mean = np.mean(self.distance_error)

        self.angle_error = np.sum(X * self.X_inverse_transform, axis=-1) / (X_lengths * X_inverse_lengths)
        self.angle_error_mean = np.mean(self.angle_error)

        self.grid_density = np.zeros(2**self.X_grid_vectors.shape[-1])
        for i in range(len(self.grid_density)):
            self.grid_density[i] = np.sum(1 * (self.X_grid_numbers == i))
        self.grid_density_mean = np.mean(self.grid_density)

        self.model_params_ready = True

        return self.model_to_b64json(numpy_to_base64_str=False)

    def fine_tune(
            self,
            X: np.ndarray,
            X_labels: list = None,
            X_full_records: list = None,
            n_components: int = None,
            # pass through mode
            test_mode = False,
    ) -> dict:
        # TODO How to fine tune?
        self.logger.warning('Fine tuning for PCA not yet implemented')
        res = self.fit(
            X = X,
            X_labels = X_labels,
            X_full_records = X_full_records,
            n_components = n_components,
        )
        return res

    # Recover estimate of original point from PCA compression
    def inverse_transform(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        try:
            self.__lock.acquire_mutexes(
                id = 'inverse_transform',
                mutexes = [self.__mutex_model],
            )
            return self.__inverse_transform(
                x_transform = X,
            )
        finally:
            self.__lock.release_mutexes(mutexes=[self.__mutex_model])

    def __inverse_transform(
            self,
            x_transform: np.ndarray,
    ) -> np.ndarray:
        self.__check_consistency(principal_components=self.model_principal_components)
        if x_transform.ndim == 1:
            x_transform = x_transform.reshape((1, x_transform.shape[0]))
        x_estimated = np.zeros(shape=(len(x_transform), self.model_centroid.shape[-1]))
        x_estimated += self.model_centroid
        # TODO how to not use loop?
        for i in range(len(x_estimated)):
            for k in range(len(self.model_principal_components)):
                x_estimated[i] += x_transform[i][k] * self.model_principal_components[k]
        return x_estimated

    # Get PCA values of arbitrary points
    def transform(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        try:
            self.__lock.acquire_mutexes(
                id = 'transform',
                mutexes = [self.__mutex_model],
            )
            return self.__transform(X=X)
        finally:
            self.__lock.release_mutexes(mutexes=[self.__mutex_model])

    def __transform(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        self.__check_consistency(principal_components=self.model_principal_components)
        assert X.shape[-1] == self.model_centroid.shape[0], (
                'Inconsistent shapes X ' + str(X.shape) + ', centroid ' + str(self.model_centroid.shape))
        X_remainder = X - self.model_centroid
        pca_coef = np.zeros(shape=(len(X), len(self.model_principal_components)))
        for k in range(len(self.model_principal_components)):
            if k > 0:
                pca_coef_1 = pca_coef[:, k - 1]
                pca_coef_1 = pca_coef_1.reshape((len(pca_coef_1), 1))
                # print('X_remainder', X_remainder)
                # print('pca coef k-1', pca_coef_1)
                # print('pca components k-1', pca_components[k-1])
                X_remainder = X_remainder - pca_coef_1 * self.model_principal_components[k - 1]
            pca_coef[:, k] = np.matmul(X_remainder, self.model_principal_components[k].transpose())
            # print(k, pca_coef)
        return pca_coef

    def __calc_pca_grid(
            self,
            x_pca: np.ndarray,
    ):
        grids = np.zeros(shape=(len(x_pca), self.model_n_components_or_centers))
        grid_numbers = np.zeros(len(x_pca))
        for i in range(self.model_n_components_or_centers):
            # Build +/- binary for each pca component. For example if we have 2 rows, with 3 PCA components
            # [
            #   [  1.2, -9.5, 0.8 ],
            #   [ -4.0, -8.5, -0.4 ],
            # ]
            # Then we would have grids
            # [
            #   [  1, -1, 1 ],
            #   [ -1, -1, -1 ],
            # ]
            # Thus n components will have 2^n grids, for the case above with 3 components will have max 8 grids.
            # Since PCA is centered, we are somewhat assured of uniform layout, and not everything concentrated
            # in a few grids.
            grids[:,i] = 1 * (x_pca[:,i] >= 0)
            # big endian
            grid_numbers += (2**(self.model_n_components_or_centers-i-1)) * grids[:,i]
        return grids, grid_numbers

    def predict(
            self,
            X: np.ndarray,
            # can be cluster numbers to zoom into
            X_search_local_space: np.ndarray = None,
            labels_search_local_space: list = None,
            top_k = 5,
            return_full_record = False,
            use_grid = False,
    ):
        try:
            self.__lock.acquire_mutexes(
                id = 'predict',
                mutexes = [self.__mutex_model],
            )

            #
            # There are 2 possible approaches, after obtaining the PCA segment numbers & relevant reference vectors:
            #    1. Transform input vector to PCA transform, then compare with the reference PCA transforms
            #    2. Do not transform, use original vectors to compare. For now, we use this approach to skip the step
            #       of transforming the input vector.
            #

            X_pca = self.__transform(
                X = X,
            )

            if use_grid:
                #
                # STEP 1: Determine the grid segments of the texts/embeddings
                #
                grid, grid_numbers = self.__calc_pca_grid(
                    x_pca = X_pca,
                )
                self.logger.info('Predict X lies in grid numbers: ' + str(grid_numbers))

                #
                # STEP 2: Get the reference text encoding, standardized labels, data records for those grid segments only
                #
                condition = False
                # recs_from_db = []
                for gn in np.unique(grid_numbers):
                    condition = condition | (self.X_grid_numbers == gn)
                # TODO This can be obtained by querying underlying DB, so we don't need to keep in RAM
                X_subset = self.X_transform[condition]
                labels_subset = np.array(self.X_labels)[condition].tolist()
                if return_full_record:
                    assert self.X_full_records is not None, 'Cannot return full records'
                    full_records_list = np.array(self.X_full_records)[condition].tolist()
                else:
                    full_records_list = None
                # data_records_subset = [r for i, r in enumerate(self.model_compression_data_records) if condition[i]]
            else:
                X_subset = self.X_transform
                labels_subset = self.X_labels
                full_records_list = self.X_full_records

            #
            # STEP 3: Call __predict()
            #
            pred_labels_or_records, pred_probs = self.predict_standard(
                X = X_pca,
                ref_X = X_subset,
                ref_labels = labels_subset,
                ref_full_records = full_records_list,
                top_k = top_k,
                return_full_record = return_full_record,
            )
            return pred_labels_or_records, pred_probs
        finally:
            self.__lock.release_mutexes(mutexes=[self.__mutex_model])

    def model_to_b64json(
            self,
            numpy_to_base64_str = False,
            dump_to_b64json_str = False,
    ):
        base_model_dict = super().model_to_b64json(
            numpy_to_base64_str = numpy_to_base64_str,
            dump_to_b64json_str = False,
        )
        if dump_to_b64json_str:
            return self.base64.encode(b=json.dumps(base_model_dict).encode(encoding='utf-8'))
        else:
            return base_model_dict

    def load_model_from_b64json(
            self,
            model_b64json,
    ):
        _ = super().load_model_from_b64json(model_b64json=model_b64json)
        return


if __name__ == '__main__':
    from fitxf.math.lang.encode.LangModelPt import LangModelPt as LmPt
    texts = [
        "Let's have coffee", "Free for a drink?", "How about Starbucks?",
        "I am busy", "Go away", "Don't disturb me",
        "Monetary policies", "Interest rates", "Deposit rates",
    ]
    lmo = LmPt(cache_folder=Env().MODELS_PRETRAINED_DIR)

    embeddings = lmo.encode(content_list=texts, return_tensors='np')

    # use the function create_pca_plot to
    fitter = FitXformPca(logger=Logging.get_default_logger(log_level=logging.INFO, propagate=False))
    pca_fit_optimal = fitter.fit_optimal(X=embeddings)
    x_fit = pca_fit_optimal['X_transform']

    fitter.create_scatter_plot2d(
        x_transform = x_fit,
        labels_list = texts,
        show = True,
    )

    x = np.array([[1,2,3], [3,2,1], [-1,-2,-2], [-3,-4,-2]])
    x_pca = fitter.fit(X=x, X_labels=['+', '+', '-', '-'], n_components=1)
    print(x_pca)
    print(fitter.predict(X=np.array([[9,9,8], [-55,-33,-55]]), use_grid=False))
    print(fitter.predict(X=np.array([[9,9,8], [-55,-33,-55]]), use_grid=True))

    exit(0)
