import logging
import os
import numpy as np
import pandas as pd
from io import StringIO
from fitxf.math.data.ut.LabelTextEmbed01 import DATA_LABEL_TEXT_EMBEDDING_01_TRAIN
from fitxf.math.algo.encoding.Base64 import Base64
from fitxf.math.fit.utils.TensorUtils import TensorUtils
from fitxf.math.fit.cluster.Metrics import Metrics as ClusterMetrics
from fitxf.math.fit.cluster.ClusterCosine import ClusterCosine
from fitxf.math.lang.encode.LangModelPtSingleton import LangModelPtSingleton, LangModelInterface
from fitxf.math.lang.encode.LangModelPt import LangModelPt
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Env import Env
from fitxf.math.utils.Pandas import Pandas


class ClusterCosineUnitTest:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.tensor_utils = TensorUtils(logger=self.logger)
        self.cluster_cos = ClusterCosine(logger=self.logger)
        self.base64 = Base64(logger=self.logger)
        return

    def test_nlp(
            self,
    ):
        er = Env()
        sample_data_csv = er.NLP_DATASET_DIR + '/lang-model-test/data.csv'
        # Force to only use pre-calculated embedding
        # is_csv_exist = os.path.exists(sample_data_csv)
        is_csv_exist = False
        self.logger.info('Test data csv exist = ' + str(is_csv_exist) + '.')

        def get_lm() -> LangModelInterface:
            return LangModelPtSingleton.get_singleton(
                LmClass = LangModelPt,
                cache_folder = er.MODELS_PRETRAINED_DIR,
                logger = self.logger,
            )

        Pandas.increase_display()
        df = pd.read_csv(
            filepath_or_buffer = sample_data_csv if is_csv_exist else StringIO(DATA_LABEL_TEXT_EMBEDDING_01_TRAIN),
            sep = ',',
            index_col = False,
        )
        columns_keep = ['label', 'text'] if is_csv_exist else ['label', 'text', 'embedding']
        for c in columns_keep: assert c in df.columns
        df = df[columns_keep]
        df.dropna(inplace=True)
        # _, _, df[self.col_label_std] = FitUtils().map_labels_to_consecutive_numbers(lbl_list=list(df[self.col_label]))
        self.logger.info('Successfully read data of shape ' + str(df.shape))
        if is_csv_exist:
            self.logger.info('Not using precalculated embeddings, recalculating text embeddings..')
            embeddings = get_lm().encode(
                content_list = df['text'].tolist(),
                return_tensors = 'np',
            )
            df['embedding'] = [
                self.base64.encode_numpy_array_to_base64_string(x=row, data_type=np.float64)
                for row in embeddings
            ]
            #df.to_csv('data_with_embedding.csv', index=False)
            #raise Exception('asdf')
        else:
            self.logger.info('Using precalculated embeddings, not calculating.')
            embeddings = np.array([
                self.base64.decode_base64_string_to_numpy_array(s64=s, data_type=np.float64)
                for s in df['embedding'].tolist()
            ])
        self.logger.info('Embeddings shape ' + str(embeddings.shape) + ', type "' + str(type(embeddings)) + '"')
        records = df.to_dict('records')
        # for i, r in enumerate(records):
        #     r['embedding'] = embeddings[i]
        #     print(i, r)

        n_unique_labels = 2 * len(pd.unique(df['label']))
        res = self.cluster_cos.kmeans(
            x = embeddings,
            n_centers = n_unique_labels,
            x_labels = df['label'].tolist(),
            km_iters = 100,
        )
        cluster_centers = np.array(res['cluster_centers'])
        point_cluster_numbers = res['cluster_labels']
        prob_cluster_no_to_labels = res['cluster_label_to_original_labels']
        self.logger.info('Probs cluster no to labels: ' + str(prob_cluster_no_to_labels))

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
        score_total = 0.
        for i, r in enumerate(records):
            c_no = point_cluster_numbers[i]
            r['cluster_number'] = c_no
            r['cluster_predicted_1'] = predicted_clusters[i, 0]
            r['cluster_predicted_2'] = predicted_clusters[i, 1]
            # Get top few predictions
            top2_label_prob = {
                k: p for i, (k, p) in enumerate(prob_cluster_no_to_labels[predicted_clusters[i, 0]].items()) if i<=1
            }
            r['pred1_label'] = list(top2_label_prob.keys())[0]
            r['pred1_prob'] = list(top2_label_prob.values())[0]
            r['pred2_label'] = list(top2_label_prob.keys())[1]
            r['pred2_prob'] = list(top2_label_prob.values())[1]
            score_total += min(1.0, 1*(r['pred1_label']==r['label']) + 0.5*(r['pred2_prob']==r['label']))

            c_pr = r['cluster_predicted_1']
            assert c_pr == c_no, \
                'For record #' + str(i) + ', text "' + str(r['text']) + '", cluster number ' \
                + str(c_no) + ' != cluster predicted ' + str(c_pr)

        score_avg = score_total / len(records)
        self.logger.info('Score average = ' + str(score_avg))
        thr_score = 0.88
        assert score_avg > thr_score, \
            'Average score in prediction ' + str(score_avg) + ' < ' + str(thr_score) \
            + ': ' + str(pd.DataFrame.from_records(records))
        print(pd.DataFrame.from_records(records))
        print('CLUSTER COSINE NLP TEST PASSED')
        return

    def test(self):
        self.test_basic()
        self.test_fine_tune()
        self.test_nlp()
        print('CLUSTER COSINE TESTS PASSED')
        return

    def test_basic(self):
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
            assert cluster_numbers.tolist() == exp_cluster_numbers, \
                    'Cluster numbers ' + str(cluster_numbers.tolist()) + ' not expected ' + str(exp_cluster_numbers)

            res = self.cluster_cos.kmeans(
                x = x,
                n_centers = n,
                km_iters = 10,
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
        self.logger.info('Basic tests passed')
        return

    def test_fine_tune(self):
        x = np.array([
            [1.0, 0.1, 0.1], [2.0, 0.2, -0.1],
            [-1.0, 1.0, 1.0], [-3.4, 2.0, 1.8],
            [-2.0, -2.3, -1.8], [-111, -100, -112],
        ])
        res = self.cluster_cos.kmeans_optimal(
            x = x,
        )
        # get first turning point
        cluster_last_turn_point = res[0]
        self.logger.info('Kmeans optimal at first turning point: ' + str(cluster_last_turn_point))
        optimal_n = cluster_last_turn_point['n_centers']
        optimal_cluster_centers = cluster_last_turn_point['cluster_centers']

        #
        # Test fine-tuning
        #
        # add new point
        x_add = np.array([[1.1, 0.2, 0.1], [-2.1, -2.0, -1.9]])
        x_new = np.append(x, x_add, axis=0)
        res = self.cluster_cos.kmeans(
            x = x_new,
            n_centers = optimal_n,
            start_centers = optimal_cluster_centers,
            km_iters = 10,
        )
        original_centers = res['cluster_centers']
        original_center_lbls = res['cluster_labels']

        # Last added point cluster number must equal 1st one
        assert original_center_lbls[-2] == original_center_lbls[0], \
            'Last added point should belong to cluster of 1st 3 points but got labels ' + str(original_center_lbls)
        assert original_center_lbls[-1] == original_center_lbls[4], \
            'Last added point should belong to cluster of 1st 3 points but got labels ' + str(original_center_lbls)
        # This is same as above, test that original clusters retained
        for i, j in [(0, 1), (2, 3), (4, 5)]:
            assert original_center_lbls[i] == original_center_lbls[j], \
                'Points ' + str((i,j)) + ' not in same cluster ' + str(original_center_lbls)

        #
        # Test fine-tuning with new cluster center, make sure n remains the same after training
        #
        # add new point
        x_add = np.array([[1.1, 0.2, 0.1], [-2.1, -2.0, -1.9]])
        x_new = np.append(x, x_add, axis=0)
        res = self.cluster_cos.kmeans(
            x = x_new,
            n_centers = optimal_n + 1,
            start_centers = np.append(original_centers, np.array([[100., -100., 100.]]), axis=0),
            km_iters = 10,
        )
        n = res['n_centers']
        clusters = res['clusters']
        new_center_lbls = res['cluster_labels']
        for clstr in clusters:
            assert len(clstr) > 0, 'Empty cluster after fine-tuning: ' + str(clusters)

        # Last added point cluster number must equal 1st one
        self.logger.info(
            'After fine tuning add another center far away, n = ' + str(n) + ', cluster labels ' + str(new_center_lbls)
            + ', clusters ' + str(clusters)
        )
        assert n == optimal_n + 1


        #
        # Test fine-tuning with new cluster center, but force it to be empty by only allowing 1 iteration
        #
        # add new point
        x_add = np.array([[1.1, 0.2, 0.1], [-2.1, -2.0, -1.9]])
        x_new = np.append(x, x_add, axis=0)
        res = self.cluster_cos.kmeans(
            x = x_new,
            n_centers = optimal_n + 1,
            start_centers = np.append(original_centers, np.array([[100., -100., 100.]]), axis=0),
            km_iters = 1,
        )
        n = res['n_centers']
        clusters = res['clusters']
        new_center_lbls = res['cluster_labels']

        # Last added point cluster number must equal 1st one
        self.logger.info(
            'After fine tuning add another center far away, n = ' + str(n) + ', cluster labels ' + str(new_center_lbls)
            + ', clusters ' + str(clusters)
        )
        assert n == optimal_n + 1
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    ClusterCosineUnitTest(
        logger = lgr,
    ).test()
    exit(0)
