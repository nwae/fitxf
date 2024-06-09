import numpy as np
import logging
from nwae.math.fit.cluster.Cluster import Cluster


class ClusterUnitTest:

    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()

    def test_converge(self):
        x = np.array([
            [5, 1, 1], [8, 2, 1], [6, 0, 2],
            [1, 5, 1], [2, 7, 1], [0, 6, 2],
            [1, 1, 5], [2, 1, 8], [0, 2, 6],
        ])
        obj = Cluster(logger=self.logger)
        res = obj.kmeans_optimal(
            x = x,
            estimate_min_max = True,
        )
        n = res[0]['n_centers']
        centers = res[0]['cluster_centers']
        center_lbls = res[0]['cluster_labels']
        obj.logger.debug('Optimal clusters = ' + str(n))
        obj.logger.debug('Cluster centers = ' + str(centers))
        obj.logger.debug('Cluster labels: ' + str(center_lbls))
        obj.logger.debug('Cluster sizes: ' + str(res[0]['cluster_sizes']))

        for i in range(len(x)):
            obs = center_lbls[i]
            exp = center_lbls[i-i%3]
            assert obs == exp, \
                'Label for index ' + str(i) + ', x = ' + str(x[i]) + ' observed ' + str(obs) + ', expected ' + str(exp)
        return

    def test_diverge(self):
        x = np.array([
            [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0],
        ])
        obj = Cluster(logger=self.logger)
        res = obj.kmeans_optimal(
            x = x,
            # allow single point clusters
            thr_single_clusters = 1.,
            estimate_min_max    = False,
            max_clusters        = len(x),
        )
        n = res[0]['n_centers']
        centers = res[0]['cluster_centers']
        center_lbls = res[0]['cluster_labels']
        obj.logger.debug('Optimal clusters = ' + str(n))
        obj.logger.debug('Cluster centers = ' + str(centers))
        obj.logger.debug('Cluster labels: ' + str(center_lbls))
        obj.logger.debug('Cluster sizes: ' + str(res[0]['cluster_sizes']))
        assert n >= 4

    def test_imbalanced(self):
        x = np.array([
            [1.0, 0, 0, 0, 0], [1.1, 0, 0, 0, 0], [0, 0, 0, 0, 1.0], [0, 0, 0, 0, 1.1], [0, 0, 0, 0, 0.9],
            [100, 0, 0, 0, 0], [101, 0, 0, 0, 0], [0, 0, 0, 0, 100], [0, 0, 0, 0, 110], [0, 0, 0, 0, 99],
        ])
        obj = Cluster(logger=self.logger)
        res = obj.kmeans_optimal(
            x = x,
            estimate_min_max = True,
        )
        # TODO Why index 1 and not 0?
        n = res[1]['n_centers']
        centers = res[1]['cluster_centers']
        center_lbls = res[1]['cluster_labels']
        obj.logger.debug('Optimal clusters = ' + str(n))
        obj.logger.debug('Cluster centers = ' + str(centers))
        obj.logger.debug('Cluster labels: ' + str(center_lbls))
        obj.logger.debug('Cluster sizes: ' + str(res[1]['cluster_sizes']))
        assert n == 4

    def test(self):
        self.test_converge()
        self.test_diverge()
        self.test_imbalanced()
        print('ALL TESTS PASSED OK')
        return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ut = ClusterUnitTest()
    ut.test()
    exit(0)
