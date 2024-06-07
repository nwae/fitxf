import numpy as np
import torch
from sklearn.cluster import KMeans
import logging
from nwae.math.fit.utils.FitUtils import FitUtils


#
# Method:
#    Given points with labels, clustering is done within the same label.
# Advantages:
#    - No need to train a grid (Voronoi tesselation, full clustering, etc.)
#    - When new point/label is added, no need to retrain grid, but just for a single label
# Disadvantages
#    - If there are many labels, compression achieved may not be significant enough
#
class FitMean:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.fit_utils = FitUtils(
            logger = self.logger,
        )
        return

    def fit_mean(
            self,
            X,          # torch.Tensor
            # 0 index, unique valus in sequence 0, 1, 2, 3, ...
            labels,     # torch.Tensor
            # if >1, we use k-cluster to get the desired number of (cluster) "means"
            mean_per_label = 1,
            normalize_X = False,
            # Clone points if not enough
            clone_points = True,
    ):
        # Don't modify input Tensor X, thus we make a copy
        X_nm = X.to(torch.float, copy=True)
        assert id(X_nm) != id(X), 'Must be different objects ' + str(id(X_nm)) + ', X ' + str(id(X))
        assert X_nm.dtype in [torch.float, torch.float64], 'Wrong type X_nm "' + str(str(X_nm.dtype)) + '"'

        if normalize_X:
            X_nm = self.fit_utils.normalize_tensor_2D(X=X)

        n_unique = len(torch.unique(labels))
        labels_unique = list(np.unique(labels))
        labels_unique.sort()
        list_0_to_n = list(range(n_unique))
        list_0_to_n.sort()
        assert labels_unique == list_0_to_n, 'Labels must be consecutive 0 to n, unique labels ' + str(labels_unique)

        mean_tensors = []
        for i_lbl in range(n_unique):
            # Indexes True/False of the desired label
            idx_tmp = labels == i_lbl
            x_tmp = X_nm[idx_tmp]
            assert x_tmp.size()[0] >= 1, 'Label ' + str(i_lbl) + ' must have at least 1 point'
            if clone_points:
                len_ori = x_tmp.size()[0]
                while x_tmp.size()[0] < mean_per_label:
                    x_tmp = torch.cat((x_tmp, x_tmp), dim=0)
                if x_tmp.size()[0] > len_ori:
                    self.logger.warning(
                        'Not enough points for label ' + str(i_lbl) + ', from ' + str(len_ori)
                        + ' points concatenated to ' + str(x_tmp.size()[0]) + ' points: ' + str(x_tmp)
                    )
            assert x_tmp.size()[0] >= mean_per_label, \
                str(x_tmp) + ' must have at least ' + str(mean_per_label) + ' points'
            # Mean of all tensors in this label, to create a single/few "representations" of the label
            if mean_per_label > 1:
                # Повторные точки могут вызывать предупреждение
                # "ConvergenceWarning: Number of distinct clusters found smaller than n_clusters."
                # Но мы это игнорируем
                # n_init должен быть достаточно большим чтобы результат сходился
                kmeans = KMeans(n_clusters=mean_per_label, n_init=10, random_state=0).fit(X=x_tmp)
                cluster_centers = kmeans.cluster_centers_
                # print(cluster_centers)
                mean_tensors.append(torch.from_numpy(cluster_centers))
            else:
                mean_tmp = x_tmp.mean(dim=0)
                mean_tensors.append(mean_tmp)
        # Create a torch tensor from a list or torch tensors
        mean_tensors = torch.stack(mean_tensors)
        # make sure is float32
        return mean_tensors.type(torch.FloatTensor)

    def predict_mean(
            self,
            X,  # torch.Tensor
            mean_tensors,   # torch.Tensor
            reps_per_label,
            top_n     = 5,
            remove_dup_lbl = True,
            normalize_X    = False,
    ):
        # Don't modify input Tensor X, thus we make a copy
        X_nm = X.to(torch.float, copy=True)
        assert id(X_nm) != id(X), 'Must be different objects ' + str(id(X_nm)) + ', X ' + str(id(X))
        assert X_nm.dtype in [torch.float, torch.float64], 'Wrong type X_nm "' + str(str(X_nm.dtype)) + '"'

        if normalize_X:
            X_nm = self.fit_utils.normalize_tensor_2D(X=X_nm)
            metric = 'dot'
        else:
            metric = 'dist'

        # Resize if there are multiple representations per label
        size_tmp = list(mean_tensors.size())
        assert len(size_tmp) in [2,3], 'Allowed dimensions [2,3] for ' + str(size_tmp)
        if len(size_tmp) == 3:
            mean_tensors_flattenned = mean_tensors.reshape(
                size_tmp[0]*size_tmp[1],
                size_tmp[2]
            )
            self.logger.info(
                'Flattened mean tensors type ' + str(mean_tensors_flattenned.dtype)
                + ' from size ' + str(mean_tensors.size()) + ' to new size ' + str(mean_tensors_flattenned.size())
            )
        else:
            mean_tensors_flattenned = mean_tensors

        count_labels = mean_tensors.size()[0]

        if metric == 'dist':
            d = torch.FloatTensor()
            # loop calculate distance of all points to each representative label, then concatenate the rows
            for i in range(mean_tensors_flattenned.size()[0]):
                s = torch.sum((X_nm - mean_tensors_flattenned[i]) ** 2, dim=-1) ** 0.5
                s = s.reshape(1, s.size()[0])
                d = torch.cat((d, s), dim=0)
            pred_y = torch.t(d)
        else:
            tmp = torch.t(mean_tensors_flattenned)
            assert tmp.dtype in [torch.float, torch.float64], 'Wrong type mean tensors "' + str(tmp.dtype) + '"'
            # If there are self.reps_per_label=3, then the correct label is floor(index/3)
            pred_y = torch.matmul(X_nm, tmp)
            # print('multiply tensors size ' + str(last_hidden_s_cls.size()) + ' with size ' + str(tmp.size()))

        if metric == 'dist':
            # min distance to max distance
            indexes_sorted = torch.argsort(pred_y, dim=-1, descending=False)
        else:
            # max dot product to min dot product
            indexes_sorted = torch.argsort(pred_y, dim=-1, descending=True)
        # If there are self.reps_per_label=3, then the correct label is floor(index/3)
        actual_labels_sorted = torch.floor(indexes_sorted / reps_per_label)

        # at this point all labels should appear the same number of times, exactly <reps_per_label> times

        # Remove duplicate labels in result if more than 1 mean_per_label
        # TODO what a mess this code, there should be a cleaner way
        if remove_dup_lbl:
            actual_labels_unique = torch.LongTensor()
            indexes_sorted_unique = torch.LongTensor()

            # Process row by row
            for i in range(actual_labels_sorted.size()[0]):
                row_actual_labels_sorted = actual_labels_sorted[i]
                row_indexes_sorted = indexes_sorted[i]

                keep_indexes = []
                keep_actual_labels = []
                for j in range(len(row_actual_labels_sorted)):
                    val = row_actual_labels_sorted[j].item()
                    idx = row_indexes_sorted[j].item()
                    if val not in keep_actual_labels:
                        keep_actual_labels.append(val)
                        keep_indexes.append(idx)
                assert len(keep_actual_labels) == count_labels, 'Length labels ' + str(len(keep_actual_labels)) + ' not ' + str(count_labels)
                assert len(keep_indexes) == count_labels, 'Length indexes ' + str(len(keep_indexes)) + ' not ' + str(count_labels)

                # Convert to torch tensor and unsqueeze to 2D
                keep_actual_labels = torch.LongTensor(keep_actual_labels).reshape(1, count_labels)
                keep_indexes = torch.LongTensor(keep_indexes).reshape(1, count_labels)
                # concatenate to final results
                actual_labels_unique = torch.cat((actual_labels_unique, keep_actual_labels), dim=0)
                indexes_sorted_unique = torch.cat((indexes_sorted_unique, keep_indexes), dim=0)
                self.logger.debug(
                    'Reduced from ' + str(row_actual_labels_sorted) + ' to ' + str(keep_actual_labels) + ', index from '
                    + str(row_indexes_sorted) + ' to ' + str(keep_indexes)
                )
            indexes_final = indexes_sorted_unique
            actual_labels_final = actual_labels_unique
        else:
            indexes_final = indexes_sorted
            actual_labels_final = actual_labels_sorted

        # Finally return the top k labels & relevant metric by indexing correctly
        probs = torch.FloatTensor()
        classes = torch.LongTensor()
        for i in range(indexes_final.shape[0]):
            p_row = pred_y[i][indexes_final[i]][:top_n]
            p_row = p_row.reshape(1, p_row.size()[0])
            c_row = actual_labels_final[i][:top_n]
            c_row = c_row.reshape(1, c_row.size()[0])
            probs = torch.cat((probs, p_row), dim=0)
            classes = torch.cat((classes, c_row), dim=0)
        return classes.type(torch.LongTensor), probs.type(torch.FloatTensor)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    fm = FitMean()
    mtsr = fm.fit_mean(
        X = torch.tensor([[1, 1], [0.5, 2], [1.5, 0.8], [-1, 1], [-0.5, 2], [-1.5, 0.8]]),
        labels = torch.tensor([0, 0, 0, 1, 1, 1]),
        mean_per_label = 2,
        normalize_X = True,
    )
    print(mtsr)

    res = fm.predict_mean(
        X = torch.tensor([[2, 3], [-3, -1]]),
        mean_tensors = mtsr,
        reps_per_label = 2,
        normalize_X = True,
    )
    print(res)
    exit(0)
