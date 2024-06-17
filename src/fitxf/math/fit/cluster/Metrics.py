import logging
import numpy as np
import pandas as pd
from fitxf.math.utils.Logging import Logging


class Metrics:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    """
    In some cases, we may have labelled data which we subject to clustering (e.g. using text embedding)
    For example:
    
           label                                               text  cluster_number
            food                                   bread and butter               3
            food                                     fish and chips               1
            food            sausages, scrambled eggs or hash browns               1
            tech                              computers and laptops               8
            tech                                   code programming               4
            tech                                    8 bits one byte               4
          sports                                 Tennis grass court               9
          sports                                   Soccer world cup               9
          sports                     50m freestyle record under 21s               7
        medicine                    Diagnosis and treatment options               7
        medicine  Changing lifestyle habits over surgery & presc...               2
        medicine        Genomic basis for RNA alterations in cancer               6
    
    Then we want to find the "purity" of the cluster numbers, whether or not each label is assigned.
    Min purity = 1.0,
    Max purity = total labels
    """
    def get_label_cluster_purity(
            self,
            point_cluster_numbers: list,
            point_labels: list,
    ):
        n_clusters = len(np.unique(point_cluster_numbers))
        assert np.max(point_cluster_numbers) == n_clusters-1
        assert np.min(point_cluster_numbers) == 0

        self.logger.info('Total unique cluster numbers ' + str(n_clusters))
        df = pd.DataFrame({'label': point_labels, 'cluster_number': point_cluster_numbers})
        # We want to extract something like this: labels --> point clusters
        #   food [3, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1, 1]
        #   tech [8, 4, 4, 8, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4]
        #   sports [9, 9, 7, 9, 9, 9, 9, 7, 9, 9, 7, 7, 9, 5]
        #   medicine [7, 2, 6, 7, 2, 5, 4, 5, 5, 5, 4, 5, 0]
        #   genetics [6, 6, 6, 6, 2, 2, 6, 2, 2, 6, 2, 2, 6, 2]
        label_to_cluster_numbers = {lbl: [] for lbl in pd.unique(df['label'])}
        # Count instead (how many in each cluster number)
        #   food [ 0 10  0  4  0  0  0  0  0  0]
        #   tech [ 0  0  0  0 11  0  0  0  3  0]
        #   sports [0 0 0 0 0 1 0 4 0 9]
        #   medicine [1 0 2 0 2 5 1 2 0 0]
        #   genetics [0 0 7 0 0 0 7 0 0 0]
        label_to_cluster_counts = {lbl: np.zeros(n_clusters, dtype=int) for lbl in pd.unique(df['label'])}
        # Same as above, but not by label, just globally
        label_to_cluster_counts_global = np.zeros(n_clusters)
        self.logger.info('Global counts: ' + str(label_to_cluster_counts_global))
        # cluster_freq_all = np.array(cluster_numbers)
        for i, r in enumerate(df.to_records()):
            c_no = r['cluster_number']
            label_to_cluster_numbers[r['label']].append(c_no)
            label_to_cluster_counts[r['label']][c_no] += 1
            label_to_cluster_counts_global[c_no] += 1
        label_to_cluster_purity = {k: v for k, v in label_to_cluster_counts.items()}
        for lbl, val in label_to_cluster_counts.items():
            # normalize such that sum is 1
            val_prob = val / np.sum(val)
            # Dividing by the global value will give us the "true" value, meaning that if for "food", it
            # is in cluster 1 ten times, and globally there are also 10 counts, then it has a pure 1.0
            # concentration.
            # So for example the data [[0,0,0,0], [1,1,1,1] will have the ownership
            # [[1.0,1.0], [1.0,1.0]] or [1,1] after multiply with prob
            # wheres the data [[0,0,1,1], [1,1,1,1]] or freq [[2,2], [0,4]]
            # will have ownership [[1,0.33],[0,0.66]] or normalized by prob to [[0.5+0.166],[0+0.666]]
            # with the 1st vector having higher "purity" of 0.
            label_to_cluster_purity[lbl] = np.sum( (val / label_to_cluster_counts_global) * val_prob )

        # [print(lbl, m) for lbl, m in label_to_cluster_numbers.items()]
        # [print(lbl, m) for lbl, m in label_to_cluster_counts.items()]
        # print('ownership', label_to_cluster_ownership)
        # print('global', label_to_cluster_counts_global)
        aggregated_purity = np.sum(np.array(list(label_to_cluster_purity.values()))) / len(label_to_cluster_purity)
        return {
            'label_purity': label_to_cluster_purity,
            'final_purity': aggregated_purity,
        }

    def map_cluster_labels_to_original_labels(
            self,
            point_cluster_numbers: list,
            point_labels: list,
    ):
        labels_unique = np.unique(point_labels).tolist()
        labels_unique.sort()
        map_index_to_labels = {i: lbl for i, lbl in enumerate(labels_unique)}
        map_labels_to_index = {v: k for k, v in map_index_to_labels.items()}
        n_labels_unique = len(labels_unique)
        # It is possible some cluster labels have no clusters, thus taking max number is more correct
        n_cno_unique = np.max(point_cluster_numbers) + 1

        self.logger.info('Total unique original labels ' + str(n_labels_unique))
        df = pd.DataFrame({'label': point_labels, 'cluster_number': point_cluster_numbers})
        self.logger.info('Label and cluster numbers: ' + str(df))
        # For example, given the following labels & cluster numbers
        #      ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
        #      [  0,   0,   1,   0,   1,   1,   0,   2,   2],
        # We want to extract something like this: point clusters --> labels
        #      0: ['a', 'a', 'b', 'c']
        #      1: ['a', 'b', 'b']
        #      2: ['c', 'c']
        clusterno_to_labelori = {cno: [] for cno in pd.unique(df['cluster_number'])}
        # Count instead (how many in each label)
        #      {0: array([2, 1, 1]), 1: array([1, 2, 0]), 2: array([0, 0, 2])}
        clusterno_to_labelori_counts = {
            cno: np.zeros(n_labels_unique, dtype=int) for cno in pd.unique(df['cluster_number'])
        }
        # Same as above, but not by label, just globally
        #       [4. 3. 2.]
        # Meaning cluster #0 appeared in 4 places, cluster #1 appeared in 3 places, and cluster #2 in 2 places
        clusterno_to_labelori_counts_global = np.zeros(n_cno_unique)
        # cluster_freq_all = np.array(cluster_numbers)
        for i, r in enumerate(df.to_records()):
            c_no = r['cluster_number']
            lbl_ori = r['label']
            clusterno_to_labelori[c_no].append(lbl_ori)
            clusterno_to_labelori_counts[c_no][map_labels_to_index[lbl_ori]] += 1
            clusterno_to_labelori_counts_global[c_no] += 1
        self.logger.info('Cluster no to original labels: ' + str(clusterno_to_labelori))
        self.logger.info('Cluster no to original labels count: ' + str(clusterno_to_labelori_counts))
        self.logger.info('Cluster no to original labels count global: ' + str(clusterno_to_labelori_counts_global))

        clusterno_to_labelori_probs = {k: v/v.sum() for k, v in clusterno_to_labelori_counts.items()}
        self.logger.info('Cluster no to original labels indexes probabilities: ' + str(clusterno_to_labelori_probs))

        # Change to more convenient form
        for k, np_prob in clusterno_to_labelori_probs.items():
            clusterno_to_labelori_probs[k] = {map_index_to_labels[i]: prob for i, prob in enumerate(np_prob)}
        self.logger.info('Cluster no to original labels probabilities: ' + str(clusterno_to_labelori_probs))

        # Probability map now looks like this
        #       {
        #          0: {'a': 0.5,   'b': 0.25,  'c': 0.25},
        #          1: {'a': 0.333, 'b': 0.666, 'c': 0.0 },
        #          2: {'a': 0.0,   'b': 0.0,   'c': 1.0 }
        #       }
        # So cluster #0 means 50% probability belongs to "a", 25% to "b", 25% to "c"
        # Cluster #2 means 33.33% "a", 66.66% "b" and 0% "c"
        # Cluster #3 means 100% sure is "c"

        # Sort a little for easier reading & easier processing when returning search results to applications
        # that can easily pick top 1 from index 0
        clusterno_to_labelori_probs_sorted = {}
        for cno, label_probs in clusterno_to_labelori_probs.items():
            df_k = pd.DataFrame({'label': label_probs.keys(), 'prob': label_probs.values()})
            df_k = df_k.sort_values(
                by = ['prob'],
                ascending = False,
            )
            clusterno_to_labelori_probs_sorted[cno] = {
                rec['label']: rec['prob'] for rec in df_k.to_dict(orient='records')
            }

        return clusterno_to_labelori_probs_sorted


if __name__ == '__main__':
    m = Metrics(logger=Logging.get_default_logger(log_level=logging.INFO, propagate=False))
    point_labels_cno = [
        # Max purity - all different
        [
            ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
            [  0,   0,   0,   1,   1,   1,   2,   2,   2],
        ],
        # Min purity, all distributed
        [
            ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
            [  0,   1,   2,   0,   1,   2,   0,   1,   2],
        ],
        # Some mixture
        [
            ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
            [  0,   0,   1,   0,   1,   1,   0,   2,   2],
        ],
    ]
    for point_labels, point_cluster_numbers in point_labels_cno:
        purity = m.get_label_cluster_purity(
            point_cluster_numbers = point_cluster_numbers,
            point_labels = point_labels,
        )
        print(purity)

        map_cno_lbl = m.map_cluster_labels_to_original_labels(
            point_cluster_numbers = point_cluster_numbers,
            point_labels = point_labels,
        )
        print(map_cno_lbl)
    exit(0)
