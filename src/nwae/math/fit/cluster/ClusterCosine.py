import logging
import numpy as np
from nwae.math.fit.cluster.Cluster import Cluster
from nwae.math.fit.utils.TensorUtils import TensorUtils
from nwae.math.utils.Logging import Logging
from nwae.math.utils.Profile import Profiling


#
# Idea is to calculate a matrix against itself for dot product
# Then we retrieve row by row, max cluster by row, until all points are covered
#
class ClusterCosine(Cluster):

    def __init__(
            self,
            logger = None,
    ):
        super().__init__(
            logger = logger,
        )
        self.tensor_utils = TensorUtils(logger=self.logger)
        self.profiler = Profiling(logger=self.logger)
        return

    def kmeans(
            self,
            x: np.ndarray,
            n_centers: int,
            x_labels: list = None,
            # for fine-tuning already trained clusters, thus no need to start all over again
            # useful for clusters of more than 1,000,000 points for example, where starting
            # again means another half day of fit training
            start_centers: np.ndarray = None,
            km_iters = 100,
            converge_diff_thr = 0.00001,
    ):
        assert km_iters > 0
        start_time = self.profiler.start()

        self.logger.info('Start cosine cos clustering x of shape ' + str(x.shape))
        # For cosine similarity, it makes no sense not to normalize first
        x_normalized = self.tensor_utils.normalize(x=x, return_tensors='np')

        centroid_move_changes = []
        if start_centers is None:
            # randomly pick n unique points
            last_centroids = np.unique(x_normalized, axis=0)[0:n_centers].tolist()
        else:
            self.logger.info('Using user provided start centers of shape ' + str(start_centers.shape))
            assert start_centers.shape[-1] == x.shape[-1], \
                'Last dim lengths not equal. Start centers shape ' + str(start_centers.shape) + ', x ' + str(x.shape)
            assert start_centers.ndim == x.ndim, \
                'Dimensions not equal, start centers shape ' + str(start_centers.shape) + ', x ' + str(x.shape)
            last_centroids = start_centers

        self.logger.debug('Initial centroids: ' + str(last_centroids))
        last_cluster_numbers = None
        last_clusters = None
        for iter in range(km_iters):
            self.logger.info('Starting iteration #' + str(iter+1) + '...')
            # Closest centers for all points
            result_ordered, mdot_ordered = self.tensor_utils.dot_sim(
                x = x_normalized,
                ref = np.array(last_centroids),
            )
            x_new_cluster_numbers = result_ordered[:,0].tolist()
            x_new_clusters = []
            for i in range(n_centers):
                i_cluster = [idx for (idx, clstr_no) in enumerate(x_new_cluster_numbers) if clstr_no == i]
                x_new_clusters.append(i_cluster)
            self.logger.debug('Result/mdot ordered ' + str(result_ordered) + ', ' + str(mdot_ordered))
            self.logger.debug('Cluster numbers: ' + str(x_new_cluster_numbers))

            # update new centroids
            updated_cluster_numbers, updated_centroids = self.get_cluster_numbers_and_centroids(
                x = x_normalized,
                clusters = x_new_clusters,
            )
            # it is easier to do Euclidean distance changes of last centers to updated centers
            dist_movements = np.sum((np.array(updated_centroids) - np.array(last_centroids)) ** 2, axis=-1) ** 0.5
            # result_ordered, mdot_ordered = self.tensor_utils.dot_sim(
            #     x = np.array(updated_centroids),
            #     ref = np.array(last_centroids),
            #     return_tensors = 'np',
            # )
            avg_dist_movements = np.mean(dist_movements)

            centroid_move_changes.append(avg_dist_movements)
            last_cluster_numbers = x_new_cluster_numbers
            last_centroids = updated_centroids
            last_clusters = x_new_clusters

            if len(centroid_move_changes) >= 2:
                move_diff = np.abs(centroid_move_changes[-2] - centroid_move_changes[-1])
                move_ratio = 100 * np.abs(centroid_move_changes[-1] / centroid_move_changes[0])
                converge_cond = move_diff < converge_diff_thr
                self.logger.info(
                    'Movement diff ' + str(move_diff) + ', move ratio with initial '  + str(move_ratio)
                    + '%. Converge condition = ' + str(converge_cond)
                )
            else:
                converge_cond = False

            self.logger.info('Done iteration #' + str(iter+1) + ', converged = ' + str(converge_cond))
            if converge_cond:
                break

        diff_secs = self.profiler.get_time_dif_secs(start=start_time, stop=self.profiler.stop(), decimals=4)
        self.logger.info(
            'Total time taken for kmeans clustering data shape ' + str(len(x_normalized.shape))
            + ' to ' + str(n_centers) + ' = ' + str(diff_secs) + 's.'
        )

        additional_info = self.derive_additional_cluster_info(
            x = x_normalized,
            n_centers = n_centers,
            cluster_centers = np.array(last_centroids),
            cluster_labels = np.array(last_cluster_numbers),
            metric = 'cosine',
        )
        if x_labels is not None:
            cluster_label_to_labelsori = self.map_centers_to_original_labels(
                labels_original = x_labels,
                labels_cluster = last_cluster_numbers,
            )
        else:
            cluster_label_to_labelsori = None
        return {
            'n_centers': n_centers,
            'clusters': last_clusters,
            'cluster_centers': np.array(last_centroids),
            # correspond to the index of the "centroids"
            'cluster_labels': np.array(last_cluster_numbers),
            'cluster_label_to_original_labels': cluster_label_to_labelsori,
            'centers_median': additional_info['centers_median'],
            'inner_radiuses': additional_info['inner_radiuses'],
            'cluster_sizes': additional_info['cluster_sizes'],
            # estimate as inner_radiuses
            'points_inertia': np.mean(additional_info['inner_radiuses']),
        }

    def get_cluster_numbers_and_centroids(
            self,
            # e.g.
            # [[1,2,0], [0,4,5], [6,3,4], [5,5,5], [0,9,8]]
            x,
            # list of clusters by x indexes e.g. [[0,1], [2,3], [4]]
            clusters: list,
    ):
        l = len(x)
        centroids = []
        cluster_numbers = np.array([-1]*l)
        for i, clstr in enumerate(clusters):
            assert len(clstr) > 0, 'Empty cluster at ' + '#' + str(i) + ', cluster ' + str(clstr)
            select = np.array([False]*l)
            for item in clstr:
                select[item] = True
            center = x[select].mean(axis=0)
            centroids.append(center.tolist())
            cluster_numbers[np.array(clstr)] = i
        return cluster_numbers, centroids

    def cluster_angle(
            self,
            # vectors
            x: np.ndarray,
            n_clusters: int,
            max_iter: int = 10,
            start_min_dist_abs: float = 0.8,
    ):
        start_time = self.profiler.start()

        x_norm = self.tensor_utils.normalize(
            x = x,
            return_tensors = 'np',
        )
        ref_norm = x_norm.transpose()
        l = len(x)
        ref_array = np.array(list(range(l)) * l).reshape(l,l) + 1

        # can be negative which means in the opposite direction from reference
        m_dot = np.matmul(x_norm, ref_norm)

        min_dist_abs = start_min_dist_abs
        clusters = []
        # we presume last move is positive, thus to tighten or increase clusters
        last_move = 0.2
        for iter in range(max_iter):
            clusters = []
            # Will give an array of 0's (fail condition) & 1's (meet condition)
            # [[1 1 0 0]
            #  [1 1 0 0]
            #  [0 0 1 1]
            #  [0 0 1 1]]
            meet_condition = 1 * (np.abs(m_dot) > np.abs(min_dist_abs))
            # Give and array of the clusters by row, 0 means None, item indexing starts from 1
            # [[1 2 0 0]
            #  [1 2 0 0]
            #  [0 0 3 4]
            #  [0 0 3 4]]
            iter_clusters = ref_array * meet_condition
            self.logger.debug(
                'Iter #' + str(iter) + ': meet condition\n' + str(meet_condition) + '\nclusters\n' + str(iter_clusters)
            )
            for i_cluster in iter_clusters:
                row_cluster = set([v for v in i_cluster if v > 0])
                self.logger.debug('Row cluster: ' + str(row_cluster))
                for existing_cluster in clusters:
                    # Keep removing items that already found clusters
                    row_cluster = set(row_cluster).difference(set(existing_cluster))
                # remaining items if non-empty & not already in clusters, then is a new cluster
                if row_cluster and (row_cluster not in clusters):
                    clusters.append(row_cluster)
            self.logger.debug('Clusters at iteration #' + str(iter) + ': ' + str(clusters))
            cur_len = len(clusters)
            if cur_len > n_clusters:
                # if last move different direction, move opposite direction a bit less, else slow down a little same dir
                move = -last_move / 2 if last_move > 0 else last_move*0.9
                # need to decrease threshold distance to reduce clusters
                min_dist_abs_new = max(0.0001, min_dist_abs + move)
                last_move = move
            elif cur_len < n_clusters:
                # if last move different direction, move opposite direction a bit less, else slow down a little same dir
                move = -last_move / 2 if last_move < 0 else last_move*0.9
                # need to increase
                min_dist_abs_new = min(0.9999, min_dist_abs + move)
                last_move = move
            else:
                break
            self.logger.info(
                'Iter #' + str(iter) + ': Adjusted dist thr from ' + str(min_dist_abs)
                + ' to new value ' + str(min_dist_abs_new) + '. Cluster n=' + str(cur_len)
                + ', target n=' + str(n_clusters) + '.'
            )
            min_dist_abs = min_dist_abs_new
        # minus 1 back
        clusters_corrected = []
        for s in clusters:
            clusters_corrected.append([element-1 for element in s])

        union_all = set()
        for s in clusters_corrected:
            union_all = union_all.union(s)
        union_all = list(union_all)
        union_all.sort()
        assert union_all == list(range(l)), 'Union all ' + str(union_all)

        cluster_numbers, centroids = self.get_cluster_numbers_and_centroids(
            x = x_norm,
            clusters = clusters_corrected,
        )

        diff_secs = self.profiler.get_time_dif_secs(start=start_time, stop=self.profiler.stop(), decimals=4)
        self.logger.info(
            'Total time taken for clustering cosine shape ' + str(len(x_norm.shape)) + ' to ' + str(n_clusters)
            + ' = ' + str(diff_secs) + 's.'
        )
        return {
            'clusters': clusters_corrected,
            # correspond to the index of the "centroids"
            'cluster_numbers': cluster_numbers.tolist(),
            # index of the centroids is the cluster numbers
            'centroids': centroids,
            'dist_threshold': min_dist_abs,
        }


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    dlen = 10000
    ddim = 384
    n = int(dlen / 100)
    x = np.random.random(size=(dlen, ddim))
    # print(x)
    ca = ClusterCosine(logger=lgr)
    # res = ca.cluster_angle(x=x, n_clusters=n, max_iter=100, start_min_dist_abs = 0.9)
    # print('------------------------------------------')
    # print(res)
    res = ca.kmeans(x=x, n_centers=n, km_iters=100)
    print('++++++++++++++++++++++++++++++++++++++++++')
    # print(res)
    print('Cluster densities: ' + str([len(c) for c in res['clusters']]))
    exit(0)
