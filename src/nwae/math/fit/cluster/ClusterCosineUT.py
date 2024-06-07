import logging
import os
import numpy as np
import pandas as pd
from nwae.math.fit.utils.TensorUtils import TensorUtils
from nwae.math.fit.cluster.Metrics import Metrics as ClusterMetrics
from nwae.math.fit.cluster.ClusterCosine import ClusterCosine
from nwae.math.lang.encode.LangModelPt import LangModelPt
from nwae.math.utils.Logging import Logging
from nwae.math.utils.EnvironRepo import EnvRepo
from nwae.math.utils.Pandas import Pandas


class ClusterCosineUnitTest:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.tensor_utils = TensorUtils(logger=self.logger)
        self.cluster_cos = ClusterCosine(logger=self.logger)
        return

    def test_nlp(self):
        er = EnvRepo(repo_dir=os.environ["REPO_DIR"])
        sample_data_csv = er.NLP_DATASET_DIR + '/lang-model-test/data.csv'

        def get_lm() -> LangModelPt:
            return LangModelPt.get_singleton(
                LmClass = LangModelPt,
                lang = 'multi',
                cache_folder = er.MODELS_PRETRAINED_DIR,
                logger = self.logger,
            )

        Pandas.increase_display()
        df = pd.read_csv(filepath_or_buffer=sample_data_csv, sep=',', index_col=False)
        columns_keep = ['label', 'text']
        for c in columns_keep: assert c in df.columns
        df = df[columns_keep]
        df.dropna(inplace=True)
        # _, _, df[self.col_label_std] = FitUtils().map_labels_to_consecutive_numbers(lbl_list=list(df[self.col_label]))
        self.logger.info('Successfully read data of shape ' + str(df.shape))
        embeddings = get_lm().encode(
            text_list = df['text'].tolist(),
            return_tensors = 'np',
        )
        records = df.to_dict('records')
        # for i, r in enumerate(records):
        #     r['embedding'] = embeddings[i]
        #     print(i, r)

        n_unique_labels = 2 * len(pd.unique(df['label']))
        res = self.cluster_cos.kmeans(
            x = embeddings,
            n_centers = n_unique_labels,
            km_iters = 100,
            init_method = 'random',
        )
        cluster_centers = np.array(res['cluster_centers'])
        point_cluster_numbers = res['cluster_labels']

        cluster_metrics = ClusterMetrics(logger=self.logger)
        purity = cluster_metrics.get_label_cluster_purity(
            point_labels = df['label'].tolist(),
            point_cluster_numbers = point_cluster_numbers,
        )
        label_purity, agg_purity = purity['label_purity'], purity['final_purity']
        self.logger.info('Label purity: ' + str(label_purity) + ', aggregated purity ' + str(agg_purity))
        min_agg_purity = 1.0 / len(label_purity)
        thr = min_agg_purity + 0.5 * (1 - min_agg_purity)
        self.logger.info('Aggregate purity ' + str(agg_purity) + ', threshold ' + str(thr))
        assert agg_purity > thr, \
            'Aggregated purity of cluster numbers ' + str(agg_purity) + ' not good enough ' + str(label_purity)

        # At this point, the application can represent each embedding with just the centroids
        # (thus cluster number as index)
        predicted_clusters, _ = self.tensor_utils.similarity_cosine(
            x = embeddings,
            ref = cluster_centers,
            return_tensors = 'np',
        )
        for i, r in enumerate(records):
            c_no = point_cluster_numbers[i]
            r['cluster_number'] = c_no
            r['cluster_predicted_1'] = predicted_clusters[i, 0]
            r['cluster_predicted_2'] = predicted_clusters[i, 1]
            c_pr = r['cluster_predicted_1']
            assert c_pr == c_no, \
                'For record #' + str(i) + ', text "' + str(r['text']) + '", cluster number ' \
                + str(c_no) + ' != cluster predicted ' + str(c_pr)

        print(pd.DataFrame.from_records(records))
        print('CLUSTER COSINE NLP TEST PASSED')
        return

    def test(self):
        x = np.array([
            [1.0, 0.1, 0.1],
            [2.0, 0.2, -0.1],
            [-1.0, 1.0, 1.0],
            [-3.4, 2.0, 1.8],
        ])
        for n, exp_clusters, exp_centroids, exp_cluster_numbers in [
            (1, [[0, 1, 2, 3]],
             [[0.1556, 0.3092, 0.2604]],
             [0, 0, 0, 0]),
            (2, [[0, 1], [2, 3]],
             [[0.9919, 0.0991, 0.0246], [-0.6807, 0.5193, 0.4962]],
             [0, 0, 1, 1]),
            (3, [[0, 1], [2], [3]],
             [[0.9919, 0.0991, 0.0246], [-0.5773, 0.5773, 0.5773], [-0.7841, 0.4612, 0.4151]],
             [0, 0, 1, 2]),
            (4, [[0], [1], [2], [3]],
             [[0.9901, 0.09901, 0.09901], [0.9938, 0.0993, -0.0496], [-0.5773, 0.5773, 0.57735], [-0.7841, 0.4612, 0.4151]],
             [0, 1, 2, 3]),
        ]:
            res = self.cluster_cos.cluster_angle(
                x = x,
                n_clusters = n,
                max_iter = 10,
                start_min_dist_abs = 0.8,
            )
            clusters = res['clusters']
            centroids = res['centroids']
            cluster_numbers = res['cluster_numbers']
            print('------------------------------------------')
            print('clustering result', res)
            print('observed', clusters, 'expected', exp_clusters)
            assert str(clusters) == str(exp_clusters), \
                'For n=' + str(n) + ', observed clusters ' + str(clusters) + ' not expected ' + str(exp_clusters)
            diff_error = np.sum( ( np.array(centroids) - np.array(exp_centroids) ) ** 2 )
            assert diff_error < 0.0000001, \
                'Observed centroids ' + str(centroids) + ' not expected ' + str(exp_centroids)
            assert cluster_numbers == exp_cluster_numbers, \
                    'Cluster numbers ' + str(cluster_numbers) + ' not expected ' + str(exp_cluster_numbers)

            res = self.cluster_cos.kmeans(
                x = x,
                n_centers = n,
                km_iters = 10,
                init_method = 'random',
            )
            print('++++++++++++++++++++++++++++++++++++++++++')
            print('By kmeans: ' + str(res))
            print(
                'Centroid changed from ' + str(centroids) + ' to ' + str(res['cluster_centers'])
                + ' Clusters: ' + str(res['clusters'])
            )

        x = np.array([
            [1.0, 0.1, 0.1], [2.0, 0.2, -0.1],
            [-1.0, 1.0, 1.0], [-3.4, 2.0, 1.8],
            [-2.0, -2.3, -1.8], [-111, -100, -112],
        ])
        res = self.cluster_cos.kmeans_optimal(
            x = x,
        )
        assert len(res) == 1, 'Expect only 1 turning point but got ' + str(len(res)) + ': ' + str(res)
        # get first turning point
        cluster_last_turn_point = res[0]
        self.logger.info('Kmeans optimal at first turning point: ' + str(cluster_last_turn_point))
        optimal_clusters = cluster_last_turn_point['clusters']
        optimal_cluster_labels = cluster_last_turn_point['cluster_labels']
        assert len(optimal_clusters) == 3, 'No of optimal clusters ' + str(len(optimal_clusters)) + ' not expected 3'
        for i, j in [(0, 1), (2, 3), (4, 5)]:
            assert optimal_cluster_labels[i] == optimal_cluster_labels[j], \
                'Points ' + str((i,j)) + ' not in same cluster ' + str(optimal_cluster_labels)

        print('Now testing via NLP..')
        self.test_nlp()
        print('CLUSTER COSINE TESTS PASSED')
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    ClusterCosineUnitTest(
        logger = lgr,
    ).test()
    exit(0)
