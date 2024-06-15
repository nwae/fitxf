import logging
import os
import numpy as np
from fitxf.math.fit.transform.FitXformInterface import FitXformInterface
from fitxf.math.fit.utils.FitUtils import FitUtils
from fitxf.math.fit.cluster.Cluster import Cluster
from fitxf.math.fit.cluster.ClusterCosine import ClusterCosine
from fitxf.math.utils.Lock import Lock
from fitxf.math.utils.EnvironRepo import EnvRepo
from fitxf.math.utils.Logging import Logging


class FitXformCluster(FitXformInterface):

    KEY_CLUSTERNO_TO_USERLABELS_INFO = 'clusterno_to_userlabels_info'

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
        self.fit_utils = FitUtils(logger=self.logger)
        self.cluster = Cluster(logger=self.logger)

        # Model parameters
        self.model_params_ready = False
        self.cluster_labels = None
        self.cluster_inertia = None
        self.cluster_inertia_per_point = None
        self.centers_median_distance = None
        return

    def is_model_ready(self):
        return self.model_params_ready

    def estimate_min_max_clusters(
            self,
            n,
    ):
        max_clusters = 3*int(np.log(n))
        min_clusters = max(2, int(np.log(n)))
        self.logger.info(
            'Min/max clusters estimated as ' + str(min_clusters) + ' and ' + str(max_clusters) + ', n=' + str(n)
        )
        return min_clusters, max_clusters

    def fit_optimal(
            self,
            X: np.ndarray,
            X_labels = None,
            X_full_records = None,
            target_grid_density = 2,
            # allowed values 'median', 'mean', 'min'
            measure = 'median',
            # Model dependent interpretation, or ignore if not relevant for specific model
            min_components = 2,
            max_components = 100,
    ) -> dict:
        try:
            self.__lock.acquire_mutexes(
                id = 'fit_optimal',
                mutexes = [self.__mutex_model],
            )

            # Interpret as min/max clusters
            min_clusters, max_clusters = min_components, max_components

            return self.__fit_optimal(
                X = X,
                X_labels = X_labels,
                X_full_records = X_full_records,
                target_grid_density = target_grid_density,
                min_clusters = min_clusters,
                max_clusters = max_clusters,
            )
        finally:
            self.__lock.release_mutexes(mutexes=[self.__mutex_model])

    def fit(
            self,
            X: np.ndarray,
            X_labels = None,
            X_full_records = None,
            # Model dependent interpretation, or ignore if not relevant for specific model
            # For example, can mean how many clusters, or how many PCA components, or how many to sample
            # in a discrete Fourier transform, etc.
            n_components = 2,
            # for fine-tuning already trained clusters, thus no need to start all over again
            # useful for clusters of more than 1,000,000 points for example, where starting
            # again means another half day of fit training
            start_centers: np.ndarray = None,
    ) -> dict:
        n_centers = n_components
        self.logger.info(
            'Start kmeans optimal with X shape ' + str(X.shape) + ', n clusters ' + str(n_centers)
        )
        desired_cluster = self.cluster.kmeans(
            x = X,
            n_centers = n_centers,
            x_labels = X_labels,
            start_centers = start_centers,
            km_iters = 100,
        )
        self.logger.info('Desired cluster of requested n=' + str(n_centers) + ': ' + str(desired_cluster))
        return self.__record_cluster(
            desired_cluster = desired_cluster,
            X = X,
            X_labels = X_labels,
            X_full_records = X_full_records,
        )

    # Will dynamically look for optimal number of pca components, on the condition of the target grid density
    def __fit_optimal(
            self,
            X: np.ndarray,
            X_labels = None,
            X_full_records = None,
            target_grid_density = 2,
            # Model dependent interpretation, or ignore if not relevant for specific model
            min_clusters = 2,
            max_clusters = 100,
            km_iters = 100,
            # by default if 25% of the clusters are single point clusters, we quit
            thr_single_clusters = 0.25,
            plot = False,
    ) -> dict:
        assert target_grid_density > 0, 'Target grid density not valid ' + str(target_grid_density)
        self.logger.info(
            'Start kmeans optimal with X shape ' + str(X.shape) + ', min clusters ' + str(min_clusters)
            + ', max clusters ' + str(max_clusters) + ', using cluster class "' + str(self.cluster.__class__) + '"'
        )
        res = self.cluster.kmeans_optimal(
            x = X,
            x_labels = X_labels,
            km_iters = km_iters,
            min_clusters = min_clusters,
            max_clusters = max_clusters,
            thr_single_clusters = thr_single_clusters,
            plot = plot,
        )
        desired_cluster = res[0]
        self.logger.info(
            'Desired optimal cluster using cluster class "' + str(self.cluster.__class__) + '": ' + str(desired_cluster)
        )
        return self.__record_cluster(
            desired_cluster = desired_cluster,
            X = X,
            X_labels = X_labels,
            X_full_records = X_full_records,
        )

    def __record_cluster(
            self,
            desired_cluster: dict,
            X: np.ndarray,
            X_labels,
            X_full_records,
    ):
        # Copy over original data
        self.X = np.array(X)
        self.X_labels = X_labels
        self.X_full_records = X_full_records
        self.model_centers = desired_cluster['cluster_centers']
        self.model_principal_components = self.model_centers
        self.cluster_labels = np.array(desired_cluster['cluster_labels'])
        self.model_n_components_or_centers = desired_cluster['n_centers']
        self.model_centroid = np.mean(self.model_centers)
        self.cluster_no_map_to_userlabel = desired_cluster["cluster_label_to_original_labels"]

        self.cluster_inertia = desired_cluster['points_inertia']
        self.cluster_inertia_per_point = self.cluster_inertia / len(self.X)
        self.distance_error = self.cluster_inertia_per_point
        self.distance_error_mean = np.mean(self.distance_error)

        self.centers_median_distance = desired_cluster['centers_median']
        self.logger.info(
            'Inertia ' + str(self.cluster_inertia) + ', inertia per point ' + str(self.cluster_inertia_per_point)
            + ', centers median distance ' + str(self.centers_median_distance)
        )

        self.X_transform = self.__transform(
            X = self.X,
        )
        self.logger.info('X transform: ' + str(self.X_transform))
        self.X_inverse_transform = self.__inverse_transform(
            x_transform = self.X_transform,
        )

        X_lengths = np.sum((self.X * self.X), axis=-1) ** 0.5
        X_inverse_lengths = np.sum((self.X_inverse_transform * self.X_inverse_transform), axis=-1) ** 0.5
        self.angle_error = np.sum(self.X * self.X_inverse_transform, axis=-1) / (X_lengths * X_inverse_lengths)
        self.angle_error_mean = np.mean(self.angle_error)

        self.X_grid_vectors = self.model_centers
        self.X_grid_numbers = self.cluster_labels

        return self.model_to_json(numpy_to_base64_str=False)

    # Inverse transform is just the cluster center
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
        # Transform is just the cluster label (or cluster center index)
        # Make sure is int type to index into model centers
        x_transform_inttype = x_transform.astype(dtype=np.int64)
        x_estimated = self.model_centers[x_transform_inttype]
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
        pred_labels, pred_probs = self.predict_standard(
            X = X,
            ref_X = self.model_centers,
            ref_labels = list(range(len(self.model_centers))),
            top_k = 3,
        )
        # Cluster transform is just the cluster label
        return np.array([r[0] for r in pred_labels])

    def __calc_grid(
            self,
            X: np.ndarray,
    ):
        raise Exception('TODO')

    def predict(
            self,
            X: np.ndarray,
            top_k = 5,
            return_full_record = False,
            use_grid = False,
    ):
        assert return_full_record == False, 'Full record return not supported for cluster prediction'
        try:
            self.__lock.acquire_mutexes(
                id = 'predict',
                mutexes = [self.__mutex_model],
            )

            pred_cluster_numbers, pred_probs = self.predict_standard(
                X = X,
                ref_X = self.model_centers,
                ref_labels = list(range(len(self.model_centers))),
                ref_full_records = self.X_full_records,
                top_k = top_k,
                return_full_record = return_full_record,
            )
            self.logger.debug('Predicted clusters for top k ' + str(top_k) + ': ' + str(pred_cluster_numbers))

            if self.cluster_no_map_to_userlabel is not None:
                # Include user label info
                pred_label_and_clusterno = []
                # map from cluster numbers to actual user labels
                for cluster_numbers in pred_cluster_numbers:
                    user_labels_cno = []
                    for cno in cluster_numbers:
                        # each cluster number is mapped to user labels probabilities, e.g.
                        #   {'food': 1.0, 'genetics': 0.0, 'medicine': 0.0, 'sports': 0.0, 'tech': 0.0}
                        tmp = self.cluster_no_map_to_userlabel[cno]
                        # we just take the top one, e.g. "food"
                        top_label_tmp = list(tmp.keys())[0]
                        user_labels_cno.append({'cluster_label': cno, 'user_label_estimate': top_label_tmp})
                    pred_label_and_clusterno.append(user_labels_cno)
                self.logger.info(
                    'Converted to user labels: ' + str(pred_label_and_clusterno)
                    + ' from cluster numbers ' + str(pred_cluster_numbers)
                )
            else:
                raise Exception('For clustering prediction, no cluster number map to user labels found')
                # pred_labels_user = pred_labels

            # Cluster transform is just the cluster label
            # return np.array([r[0] for r in pred_labels])
            return pred_label_and_clusterno, pred_probs
        finally:
            self.__lock.release_mutexes(mutexes=[self.__mutex_model])

    def model_to_json(
            self,
            numpy_to_base64_str = False,
    ):
        base_model_json = super().model_to_json(numpy_to_base64_str=numpy_to_base64_str)
        # Add more info
        base_model_json[self.KEY_CLUSTERNO_TO_USERLABELS_INFO] = self.cluster_no_map_to_userlabel
        return base_model_json

    def load_model_from_json(
            self,
            model_json,
    ):
        super().load_model_from_json(model_json=model_json)
        self.cluster_no_map_to_userlabel = model_json[self.KEY_CLUSTERNO_TO_USERLABELS_INFO]
        return


class FitXformClusterCosine(FitXformCluster):

    def __init__(
            self,
            logger = None,
    ):
        super().__init__(
            logger = logger,
        )
        # Overwrite to use ClusterCosine
        self.cluster = ClusterCosine(logger=self.logger)
        return


if __name__ == '__main__':
    from fitxf.math.lang.encode.LangModelPt import LangModelPt as LmPt
    texts = [
        "Let's have coffee", "Free for a drink?", "How about Starbucks?",
        "I am busy", "Go away", "Don't disturb me",
        "Monetary policies", "Interest rates", "Deposit rates",
    ]
    lmo = LmPt(lang='en', cache_folder=EnvRepo(repo_dir=os.environ.get("REPO_DIR", None)).MODELS_PRETRAINED_DIR)

    embeddings = lmo.encode(text_list=texts, return_tensors='np')

    # use the function create_pca_plot to
    fitter = FitXformCluster(logger=Logging.get_default_logger(log_level=logging.INFO, propagate=False))
    x_compressed = fitter.fit_optimal(X=embeddings)

    fitter.create_scatter_plot2d(
        x_transform = fitter.X_inverse_transform,
        labels_list = texts,
        show = True,
        # add some noise to separate the points a little more
        add_noise = True,
    )

    x = np.array([[1,2,3], [3,2,1], [-1,-2,-2], [-3,-4,-2]])
    res = fitter.fit(X=x, X_labels=['+', '+', '-', '-'], n_components=2)
    print('Cluster info:')
    [print(k,v) for k,v in res.items()]
    print('Fit arbitrary:', fitter.predict(X=np.array([[9,9,8], [-55,-33,-55]]), use_grid=False))
    print('Fit arbitrary:', fitter.predict(X=np.array([[9,9,8], [-55,-33,-55]]), use_grid=True))

    exit(0)
