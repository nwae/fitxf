import logging
import numpy as np
from fitxf.math.fit.cluster.Cluster import Cluster
from fitxf.math.fit.utils.TensorUtils import TensorUtils
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Profile import Profiling


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
            # pass through mode
            test_mode = False,
    ):
        if test_mode:
            raise Exception('Test mode not supported')

        assert km_iters > 0
        start_time = self.profiler.start()

        if x_labels is None:
            x_labels = list(range(len(x)))

        self.logger.info('Start cosine cos clustering x of shape ' + str(x.shape))
        # For cosine similarity, it makes no sense not to normalize first
        x_normalized = self.tensor_utils.normalize(x=x, return_tensors='np')

        centroid_move_changes = []
        if start_centers is None:
            # randomly pick n unique points
            last_centers = np.unique(x_normalized, axis=0)[0:n_centers]
        else:
            self.logger.info('Using user provided start centers of shape ' + str(start_centers.shape))
            assert start_centers.shape[-1] == x.shape[-1], \
                'Last dim lengths not equal. Start centers shape ' + str(start_centers.shape) + ', x ' + str(x.shape)
            assert start_centers.ndim == x.ndim, \
                'Dimensions not equal, start centers shape ' + str(start_centers.shape) + ', x ' + str(x.shape)
            assert len(start_centers) == n_centers, \
                'N centers ' + str(n_centers) + ' not same length with start centers ' + str(start_centers.shape)
            last_centers = start_centers

        last_cluster_numbers = None
        last_clusters = None
        total_iters = 0
        for iter in range(km_iters):
            self.logger.info('Starting iteration #' + str(iter+1) + '...')
            # Closest centers for all points
            result_ordered, mdot_ordered = self.tensor_utils.dot_sim(
                x = x_normalized,
                ref = last_centers,
            )
            x_new_cluster_numbers = result_ordered[:,0].tolist()
            x_new_clusters = []
            for i in range(n_centers):
                i_cluster = [idx for (idx, clstr_no) in enumerate(x_new_cluster_numbers) if clstr_no == i]
                x_new_clusters.append(i_cluster)
            self.logger.debug('Result/mdot ordered ' + str(result_ordered) + ', ' + str(mdot_ordered))
            self.logger.debug(
                'N centers required ' + str(n_centers) + ', cluster numbers: ' + str(x_new_cluster_numbers)
            )

            # update new centroids
            updated_cluster_numbers, updated_centers = self.get_cluster_numbers_and_centroids(
                x = x_normalized,
                clusters = x_new_clusters,
            )
            assert updated_cluster_numbers.tolist() == x_new_cluster_numbers, \
                'Consistency off from updated cluster numbers ' + str(list(zip(updated_cluster_numbers, x_new_cluster_numbers)))
            assert updated_centers.shape == last_centers.shape, \
                'Centers shape not same after update ' + str(updated_centers.shape) + ' != ' + str(last_centers.shape)
            # it is easier to do Euclidean distance changes of last centers to updated centers
            dist_movements = np.sum((updated_centers - last_centers) ** 2, axis=-1) ** 0.5
            avg_dist_movements = np.mean(dist_movements)
            centroid_move_changes.append(avg_dist_movements)


            last_cluster_numbers = updated_cluster_numbers
            last_centers = updated_centers
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
            total_iters = iter
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
            cluster_centers = last_centers,
            cluster_labels = last_cluster_numbers,
            metric = 'cosine',
        )
        if x_labels is not None:
            cluster_label_to_labelsori = self.map_centers_to_original_labels(
                labels_original = x_labels,
                labels_cluster = last_cluster_numbers.tolist(),
            )
        else:
            cluster_label_to_labelsori = None
        return {
            'total_iterations': total_iters,
            'n_centers': n_centers,
            'clusters': last_clusters,
            'cluster_centers': last_centers,
            # correspond to the index of the "centroids"
            'cluster_labels': last_cluster_numbers,
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
        len_x = len(x)
        center_shape = list(x.shape[1:])
        self.logger.info(
            'Clusters len ' + str(len(clusters)) + ', x shape ' + str(x.shape)
            + ', center without first dim shape ' + str(center_shape)
        )
        center_shape_1d = 1
        for i in center_shape:
            center_shape_1d *= i

        new_centers = []
        empty_centers_indexes = []
        cluster_item_counts = []
        cluster_numbers = np.array([-1]*len_x)
        for i, clstr in enumerate(clusters):
            # assert len(clstr) > 0, 'Empty cluster at ' + '#' + str(i) + ', cluster ' + str(clstr)
            # It can happen that no points are in cluster i, when for example user provided the start centers
            # during iteration, thus we must simply pick another suitable random point
            if len(clstr) == 0:
                self.logger.warning('No points attached to cluster center i=' + str(i))
                empty_centers_indexes.append(i)
                new_centers.append(None)
                cluster_item_counts.append(0)
            else:
                select = np.array([False]*len_x)
                for item in clstr:
                    select[item] = True
                center = x[select].mean(axis=0)
                new_centers.append(center.tolist())
                cluster_numbers[np.array(clstr)] = i
                cluster_item_counts.append(len(clstr))

        if len(empty_centers_indexes) > 0:
            self.logger.info(
                'Empty clusters exist total ' + str(len(empty_centers_indexes))
                + ' at center indexes ' + str(empty_centers_indexes)
            )
            np_cluster_item_counts = np.array(cluster_item_counts)
            # Now we know which index has most points (last index), and which has least (index 0)
            np_cluster_item_counts_sorted = np.argsort(np_cluster_item_counts)
            # We split the clusters with most points if there are empty clusters
            clusterno_to_split = np_cluster_item_counts_sorted[-1]
            cluster_to_split = clusters[clusterno_to_split]
            self.logger.info(
                'Empty clusters during train, using cluster at index #' + str(clusterno_to_split) + ' to split with '
                + str(len(cluster_to_split)) + ' elements ' + str(cluster_to_split)
            )

            for i in empty_centers_indexes:
                j = np.random.randint(low=0, high=len(cluster_to_split), size=1)[0]
                point_index = cluster_to_split[j]
                random_new_center = np.reshape(x[point_index], newshape=center_shape)
                # Make sure add a very small random number
                random_new_center += np.reshape(np.random.rand(center_shape_1d), newshape=center_shape) / 1000000
                new_centers[i] = random_new_center.tolist()
                self.logger.info(
                    'Reassigning empty cluster center ' + str(i) + ' a random point taking mean from index '
                    + str(point_index) + '.'
                )
        self.logger.info('Final centers length ' + str(len(new_centers)))
        self.logger.debug('Final clusters ' + str(cluster_numbers))
        return np.array(cluster_numbers), np.array(new_centers)

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
            # minus 1 and also convert back to int from numpy type int32 or int64
            clusters_corrected.append([int(element-1) for element in s])

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
        self.logger.info(
            'Done cluster: ' + str(clusters_corrected) + ', cluster numbers: ' + str(cluster_numbers)
            + ', centroids: ' + str(centroids)
        )

        diff_secs = self.profiler.get_time_dif_secs(start=start_time, stop=self.profiler.stop(), decimals=4)
        self.logger.info(
            'Total time taken for clustering cosine shape ' + str(len(x_norm.shape)) + ' to ' + str(n_clusters)
            + ' = ' + str(diff_secs) + 's.'
        )
        return {
            'clusters': clusters_corrected,
            # correspond to the index of the "centroids"
            'cluster_numbers': cluster_numbers,
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
