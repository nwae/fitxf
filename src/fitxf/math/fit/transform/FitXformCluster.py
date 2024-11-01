import logging
import os
import json
import pickle
import numpy as np
from fitxf.math.fit.transform.FitXformInterface import FitXformInterface
from fitxf.math.fit.utils.FitUtils import FitUtils
from fitxf.math.fit.cluster.Cluster import Cluster
from fitxf.math.fit.cluster.ClusterCosine import ClusterCosine
from fitxf.math.utils.Lock import Lock
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Env import Env


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
            X_labels: list = None,
            X_full_records: list = None,
            target_grid_density = 2,
            # allowed values 'median', 'mean', 'min'
            measure = 'median',
            # Model dependent interpretation, or ignore if not relevant for specific model
            min_components = 2,
            max_components = 100,
            test_mode = False,
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
                test_mode = test_mode,
            )
        finally:
            self.__lock.release_mutexes(mutexes=[self.__mutex_model])

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
        n_centers = n_components
        if X_labels is None:
            X_labels = list(range(len(X)))

        self.logger.info(
            'Start kmeans optimal with X shape ' + str(X.shape) + ', n clusters ' + str(n_centers)
            + ', test mode ' + str(test_mode)
        )
        desired_cluster = self.cluster.kmeans(
            x = X,
            n_centers = n_centers,
            x_labels = X_labels,
            start_centers = start_centers,
            km_iters = 100,
            test_mode = test_mode,
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
            X_labels: list = None,
            X_full_records: list = None,
            target_grid_density = 2,
            # Model dependent interpretation, or ignore if not relevant for specific model
            min_clusters = 2,
            max_clusters = 100,
            km_iters = 100,
            # by default if 25% of the clusters are single point clusters, we quit
            thr_single_clusters = 0.25,
            plot = False,
            # pass through mode
            test_mode = False,
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
            test_mode = test_mode,
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

    def fine_tune(
            self,
            X: np.ndarray,
            X_labels: list = None,
            X_full_records: list = None,
            n_components: int = None,
            # pass through mode
            test_mode = False,
    ) -> dict:
        if self.model_n_components_or_centers is not None:
            n_add = n_components - self.model_n_components_or_centers
            if n_add > 0:
                centers_additional = np.random.rand(n_add, self.model_centers.shape[-1])
                start_centers = np.append(self.model_centers, centers_additional, axis=0)
            elif n_add == 0:
                start_centers = self.model_centers
            else:
                start_centers = self.model_centers[:n_add]
            start_centers_shape = start_centers.shape
        else:
            # set as nothing to add to additional model (which does not exist)
            n_add = 0
            # if None means will train from scratch
            start_centers = None
            start_centers_shape = None

        self.logger.info(
            'Start fine tuning by additional ' + str(n_add) + ' centers, shape of start centers '
            + str(start_centers_shape) + '.'
        )
        res = self.fit(
            X = X,
            X_labels = X_labels,
            X_full_records = X_full_records,
            n_components = n_components,
            start_centers = start_centers,
            test_mode = test_mode,
        )
        return res

    def __record_cluster(
            self,
            X: np.ndarray,
            desired_cluster: dict,
            X_labels,
            X_full_records,
    ):
        # Copy over original data, don't retain large tensors in memory like X
        # self.X = np.array(X)
        self.X_labels = X_labels
        self.X_full_records = X_full_records
        self.model_train_total_iterations = desired_cluster.get('total_iterations', None)
        self.model_centers = desired_cluster['cluster_centers']
        # not applicable to cluster model
        self.model_principal_components = np.array([])
        self.cluster_labels = desired_cluster['cluster_labels']
        self.model_n_components_or_centers = desired_cluster['n_centers']
        self.model_centroid = np.mean(self.model_centers)
        self.cluster_no_map_to_userlabel = desired_cluster["cluster_label_to_original_labels"]

        self.cluster_inertia = desired_cluster['points_inertia']
        self.cluster_inertia_per_point = self.cluster_inertia / len(X_labels)
        self.distance_error = self.cluster_inertia_per_point
        self.distance_error_mean = np.mean(self.distance_error)

        self.centers_median_distance = desired_cluster['centers_median']
        self.logger.info(
            'Inertia ' + str(self.cluster_inertia) + ', inertia per point ' + str(self.cluster_inertia_per_point)
            + ', centers median distance ' + str(self.centers_median_distance)
        )

        # Same as cluster labels
        # self.X_transform = self.__transform(X=X)   # also can use transform
        self.X_transform = np.array(self.cluster_labels, dtype=np.int64)
        # assert self.X_transform.tolist() == self.cluster_labels, \
        #     'Inconsistent ' + str(list(zip(self.X_transform.tolist(), self.cluster_labels.tolist())))

        self.logger.info('X transform: ' + str(self.X_transform))
        self.X_inverse_transform = self.__inverse_transform(
            x_transform = self.X_transform,
        )

        X_lengths = np.sum((X * X), axis=-1) ** 0.5
        X_inverse_lengths = np.sum((self.X_inverse_transform * self.X_inverse_transform), axis=-1) ** 0.5
        self.angle_error = np.sum(X * self.X_inverse_transform, axis=-1) / (X_lengths * X_inverse_lengths)
        self.angle_error_mean = np.mean(self.angle_error)

        # not applicable to cluster model
        self.X_grid_vectors = np.array([])
        self.X_grid_numbers = np.array([])

        self.model_params_ready = True

        return self.model_to_b64json(
            numpy_to_base64_str = False,
            dump_to_b64json_str = False,
        )

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
            # can be cluster numbers to zoom into
            X_search_local_space: np.ndarray = None,
            labels_search_local_space: list = None,
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

            if X_search_local_space is None:
                # use model centers if local space not specified
                X_ref = self.model_centers
                # labels are just the indexes of the centers - cluster labels
                labels_ref = list(range(len(self.model_centers)))
                using_cluster_labels = True
            else:
                X_ref = X_search_local_space
                # user provided labels
                labels_ref = labels_search_local_space
                using_cluster_labels = False
                assert labels_ref is not None
                assert len(X_ref) == len(labels_ref), \
                    'Inconsistent lengths, X_search_local_space shape ' + str(X_search_local_space.shape) \
                    + ', labels_search_local_space length ' + str(len(labels_search_local_space))

            pred_labels, pred_probs = self.predict_standard(
                X = X,
                ref_X = X_ref,
                ref_labels = labels_ref,
                ref_full_records = self.X_full_records,
                top_k = top_k,
                return_full_record = return_full_record,
            )
            self.logger.debug('Predicted clusters for top k ' + str(top_k) + ': ' + str(pred_labels))

            if using_cluster_labels:
                assert self.cluster_no_map_to_userlabel is not None
                # Include user label info
                pred_label_and_clusterno = []
                # map from cluster numbers to actual user labels
                self.logger.debug(
                    'Try map cluster numbers ' + str(pred_labels) + ' to actual user labels using map: '
                    + str(self.cluster_no_map_to_userlabel)
                )
                for cluster_numbers in pred_labels:
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
                    + ' from cluster numbers ' + str(pred_labels)
                )

                # Do a 2nd zoom search
                # condition_same_cluster = self.X_transform == top_cluster
                # To simplify matters, we leave it to the caller to pull required data and call this
                # function again passing in X_search_local_space & labels_search_local_space
            else:
                pred_label_and_clusterno = pred_labels
                # pred_labels_user = pred_labels

            # Cluster transform is just the cluster label
            # return np.array([r[0] for r in pred_labels])
            return pred_label_and_clusterno, pred_probs
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
        # Add more info
        # Keep dictionary as string, because json cannot dump or load np.int32, and that problem is hard
        # to trace where the int32 comes from, so keep the whole dict as string and load back later by literal_eval
        base_model_dict[self.KEY_CLUSTERNO_TO_USERLABELS_INFO] = self.base64.encode(
            b = pickle.dumps(self.cluster_no_map_to_userlabel),
        )
        if dump_to_b64json_str:
            # retain UTF-8 if any
            return self.base64.encode(b=json.dumps(base_model_dict, ensure_ascii=False).encode(encoding='utf-8'))
        else:
            return base_model_dict

    def load_model_from_b64json(
            self,
            model_b64json,
    ) -> dict:
        model_dict = super().load_model_from_b64json(
            model_b64json = model_b64json,
        )
        self.logger.info('Loaded model dict from base class as: ' + str(model_dict))
        [self.logger.info('Key "' + str(k) + '": Value ' + str(v)) for k, v in model_dict.items()]

        # Load back dict using literal_eval, to avoid int32 json problems
        self.cluster_labels = list(self.X_transform)
        self.cluster_no_map_to_userlabel = pickle.loads(
            self.base64.decode(s=model_dict[self.KEY_CLUSTERNO_TO_USERLABELS_INFO]),
        )
        model_dict[self.KEY_CLUSTERNO_TO_USERLABELS_INFO] = self.cluster_no_map_to_userlabel
        self.logger.info(
            'Loaded model from saved base64-json string, for cluster class "' + str(self.cluster.__class__)
            + ' data length ' + str(len(self.X_transform))
            + ', n components/centers ' + str(self.model_n_components_or_centers)
            + ', centers ' + str(len(self.model_centers))
            + ', model base64-json string: ' + str(model_b64json)
        )
        return model_dict


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
    labels = ['coffee', 'coffee', 'coffee', 'busy', 'busy', 'busy', 'ir', 'ir', 'ir']
    lmo = LmPt(cache_folder=Env().MODELS_PRETRAINED_DIR)

    embeddings = lmo.encode(content_list=texts, return_tensors='np')

    fitter = FitXformCluster(logger=Logging.get_default_logger(log_level=logging.INFO, propagate=False))
    res = fitter.fit_optimal(
        X = embeddings,
        X_labels = labels,
        test_mode = True,
    )
    # Save model to base64-json string
    model_save_str = fitter.model_to_b64json(numpy_to_base64_str=True, dump_to_b64json_str=True)

    # Load from saved model
    fitter2 = FitXformCluster(logger=Logging.get_default_logger(log_level=logging.INFO, propagate=False))
    fitter2.load_model_from_b64json(model_b64json=model_save_str)
    embeddings_clusterno = fitter2.cluster_labels
    print('Embeddings cluster labels ' + str(embeddings_clusterno))

    print('Cluster info:')
    [print(k,v) for k,v in res.items()]
    pred_cnos, pred_probs = fitter2.predict(
        X  = lmo.encode(content_list=['latte with cream']),
        use_grid = False
    )
    top_cluster = pred_cnos[0][0]['cluster_label']
    print('Search among centers: ', top_cluster, pred_cnos, pred_probs)

    # Zoom search in local space
    cond = [True if cno==top_cluster else False for cno in embeddings_clusterno]
    print(cond)
    print('Search among cluster "' + str(top_cluster) + '":', fitter2.predict(
        X = lmo.encode(content_list=['latte with cream']),
        X_search_local_space = embeddings[np.array(cond)],
        labels_search_local_space = [l for i,l in enumerate(labels) if cond[i]==True],
        use_grid = False
    ))
    exit(0)

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
