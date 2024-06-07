import logging
import numpy as np
import pandas as pd
from nwae.math.utils.Logging import Logging


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
    ]
    for point_labels, point_cluster_numbers in point_labels_cno:
        purity = m.get_label_cluster_purity(
            point_cluster_numbers = point_cluster_numbers,
            point_labels = point_labels,
        )
        print(purity)
    exit(0)
