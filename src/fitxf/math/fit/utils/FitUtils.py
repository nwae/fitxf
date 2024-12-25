import logging
import numpy as np
import torch
import matplotlib.pyplot as mplt
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from fitxf.math.fit.utils.TensorUtils import TensorUtils
from fitxf.math.utils.Profile import Profiling


class FitUtils:

    @staticmethod
    def map_labels_to_consecutive_numbers(
            lbl_list,
    ):
        map_lbl_to_idx = FitUtils().map_labels_to_numbers(
            lbl_list = lbl_list,
            existing_label_map = None,
        )
        map_idx_to_lbl = {i:lbl for lbl,i in map_lbl_to_idx.items()}
        lbl_list_mapped = [map_lbl_to_idx[lbl] for lbl in lbl_list]
        return map_idx_to_lbl, map_lbl_to_idx, lbl_list_mapped

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.tensor_utils = TensorUtils(logger=self.logger)
        return

    def map_labels_to_numbers(
            self,
            lbl_list,
            existing_label_map = None,
            strict_1_to_1 = False,
    ):
        existing_label_map = {} if existing_label_map is None else existing_label_map

        existing_lbls = set(existing_label_map.keys())
        existing_idxs = set(existing_label_map.values())
        not_1_to_1 = len(existing_lbls) != len(existing_idxs)
        if not_1_to_1:
            errmsg = 'Unique labels (' + str(len(existing_lbls)) + ') != unique indexes (' + str(len(existing_idxs)) \
                     + ') in existing Label map ' + str(existing_label_map)
            self.logger.error(errmsg)
            if strict_1_to_1:
                raise Exception(errmsg)

        n_unique = len(np.unique(lbl_list + list(existing_lbls)))

        new_unique_lbls = list( set(lbl_list).difference(existing_lbls) )
        new_unique_lbls.sort()
        fill_indexes = list( set(range(n_unique)).difference(existing_idxs) )
        fill_indexes.sort()
        fill_indexes = fill_indexes[0:len(new_unique_lbls)]

        list_map_lbl_all = list(existing_label_map.items()) + list(zip(new_unique_lbls, fill_indexes))
        list_map_lbl_all.sort()
        self.logger.debug(
            'Existing labels: ' + str(existing_lbls) + ', new labels: ' + str(new_unique_lbls)
            + ', existing indexes: ' + str(existing_idxs) + ', new indexes: ' + str(fill_indexes)
            + ', list map lbl ' + str(list_map_lbl_all)
        )
        return {k: v for k, v in list_map_lbl_all}

    def get_point_distances(
            self,
            np_tensors: np.ndarray,
            # if not None, we take distances of points relative to this
            np_center = None,
            # permitted values 'exact', 'mc'
            method = 'exact',
            # permitted values 'euclid', 'euclid-square', 'cosine'
            metric = 'euclid',
            mc_iters = 10000,
    ):
        assert metric in ['euclid', 'euclid-square', 'cosine'], 'Metric not allowed "' + str(metric) + '"'
        # do a Monte-carlo
        distances = []
        n_points = len(np_tensors)
        if np_center is None:
            assert n_points >= 2, 'At least 2 points required for distances relative to each other'
        else:
            assert n_points >= 1, 'At least 1 point required for distances relative to center'

        self.logger.debug('Find by method "' + str(method) + '" distances between ' + str(n_points) + ' points')

        # Select pair indexes first, without calculating distance yet
        index_pairs = []
        if method == 'mc':
            while True:
                if len(index_pairs) >= mc_iters:
                    break
                a, b = np.random.randint(low=0, high=n_points, size=2)
                if a != b:
                    index_pairs.append((a, b))
        else:
            if np_center is None:
                for i in range(n_points):
                    [index_pairs.append((i, k)) for k in range(i + 1, n_points, 1)]
                assert len(index_pairs) == n_points * (n_points - 1) / 2
            else:
                [index_pairs.append((i, i)) for i in range(len(np_tensors))]

        self.logger.debug('Index pairs: ' + str(index_pairs))

        for a, b in index_pairs:
            if metric in ['euclid', 'euclid-square']:
                np_ref = np_tensors[b] if np_center is None else np_center
                euclid_square = np.sum( (np_tensors[a] - np_ref)**2 )
                distances.append(euclid_square) if metric == 'euclid-square' else distances.append(euclid_square ** 0.5)
            elif metric in ['cosine']:
                np_ref = np_tensors[b:(b+1)] if np_center is None else np_center.reshape([1]+list(np_center.shape))
                top_indexes, cos_dist = self.tensor_utils.dot_sim(
                    x = np_tensors[a:(a+1)],
                    ref = np_ref,
                    return_tensors = 'np',
                )
                self.logger.debug(
                    'Result cosine distance between index ' + str(a) + ' and index ' + str(b) + ', got ' + str(cos_dist)
                )
                distances.append(cos_dist)
            else:
                raise Exception('Unsupported metric "' + str(metric) + '"')
        return np.array(distances)

    def torch_train(
            self,
            model,
            loss_func,
            optimizer,
            # just an array of batch tuples, not necessarily the stupid overcomplicated torch DataLoader
            #    e.g. [(X_batch_1, y_batch_1), (X_batch_2, y_batch_2), ...]
            # or if attention masks included
            #    e.g. [(X_batch_1, attn_mask_batch_1, y_batch_1), (X_batch_2, attn_mask_batch_2, y_batch_2), ...]
            train_dataloader,
            epochs,
            regularization_type = None,
            regularization_lambda = 0.01,
            accum_steps = 1,
            plot_losses = False,
    ):
        profiler = Profiling(logger=self.logger)
        profiler_id = 'model=' + str(model) + ', epochs=' + str(epochs) + ', accum_steps=' + str(accum_steps)
        profiler.start_time_profiling(id=profiler_id)
        # Train mode means dropout layers are activated, gradients, etc
        model.train()
        losses = []
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print_epochs = np.round( np.linspace(0, epochs-1, 5), 0 )
        for epoch in range(epochs):
            run_loss = self.torch_train_1_epoch(
                epoch = epoch,
                dataloader = train_dataloader,
                model = model,
                loss_func = loss_func,
                regularization_type = regularization_type,
                regularization_lambda = regularization_lambda,
                optimizer = optimizer,
                accum_steps = accum_steps,
            )
            losses.append(run_loss)
            if epoch in print_epochs:
                pct = round(100 * epoch / (epochs-1), 0)
                self.logger.info(
                    str(pct) + '% done, losses over ' + str(epoch) + ' epochs, from ' + str(losses[0])
                    + ' to ' + str(losses[-1])
                )

        self.logger.debug('Losses over ' + str(epochs) + ' epochs: ' + str(losses))
        profiler.record_time_profiling(id=profiler_id, msg=profiler_id, logmsg=True)
        if plot_losses:
            mplt.plot(losses, linestyle='dotted')
            mplt.show()
        return losses

    def torch_train_1_epoch(
            self,
            epoch,
            # torch DataLoader
            dataloader,
            model,
            loss_func,
            optimizer,
            regularization_type = None,
            regularization_lambda = 0.01,
            # gradient accumulation steps
            # see https://wandb.ai/wandb_fc/tips/reports/How-To-Implement-Gradient-Accumulation-in-PyTorch--VmlldzoyMjMwOTk5
            accum_steps = 1,
    ):
        running_loss = 0.
        # Total step/batch should be int(n_train_samples/batch_size)
        for step, batch in enumerate(dataloader, 0):
            # batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            start_accum = (step % accum_steps == 0)
            done_accum = ((step + 1) % accum_steps == 0) or (step + 1 == len(dataloader))

            if start_accum:
                # reset model paramater gradients before forward pass
                optimizer.zero_grad()

            # Forward pass
            pred_batch_y = model(b_input_ids)
            loss = loss_func(pred_batch_y, b_labels)
            loss = loss / accum_steps

            # Regularization
            if type(regularization_type) in [int, float]:
                reg_pwr = regularization_type
            # Apply L1 regularization
            elif regularization_type == 'L1':
                reg_pwr = 1
            # Apply L2 regularization
            elif regularization_type == 'L2':
                reg_pwr = 2
            else:
                reg_pwr = 0

            if reg_pwr > 0:
                l_norm = sum(p.abs().pow(reg_pwr).sum() for p in model.parameters())
                reg_penalty = regularization_lambda * l_norm
            else:
                reg_penalty = 0
            self.logger.debug(
                'Regularization "' + str(regularization_type) + '" regularization penalty ' + str(reg_penalty)
                + ', loss from ' + str(loss) + ' to new loss ' + str(loss+reg_penalty) + ', model parameters: '
                + str([p for p in model.parameters()])
            )
            loss += reg_penalty

            # Backward pass
            loss.backward()

            if done_accum:
                # adjust model parameter weights
                optimizer.step()

            running_loss += loss.item() * accum_steps
            self.logger.debug(
                'Epoch ' + str(epoch) + ': step ' + str(step) + ' loss ' + str(running_loss / 2000)
            )
        return running_loss

    def training_data_split(
            self,
            val_ratio,
            inputs,                 # torch.Tensor
            labels,                 # torch.Tensor
            attn_masks     = None,  # torch.Tensor
            # if False, means won't care if some labels only have 1 data
            stratify_label = False,
            return_tensors = 'pt',
            return_indexes = False,
    ):
        if val_ratio <= 0:
            # Return whole data set for both training/validation
            train_idx = np.arange(len(labels))
            val_idx = np.arange(len(labels))
        else:
            # Indices of the train and validation splits stratified by labels
            train_idx, val_idx = train_test_split(
                # We just need to return indexes of split, so just pass an array of 0 index to size of labels
                np.arange(len(labels)),
                test_size = val_ratio,
                shuffle   = True,
                # if None, means won't care if some labels only have 1 data
                stratify  = None if not stratify_label else labels,
                # stratify  = labels
            )
        # Train and validation sets
        if attn_masks is None:
            fake_mask = torch.ones(len(inputs[train_idx]))
            train_set = TensorDataset(inputs[train_idx], fake_mask, labels[train_idx])
            val_set = TensorDataset(inputs[val_idx], fake_mask, labels[val_idx])
        else:
            train_set = TensorDataset(inputs[train_idx], attn_masks[train_idx], labels[train_idx])
            val_set = TensorDataset(inputs[val_idx], attn_masks[val_idx], labels[val_idx])
        if return_indexes:
            return train_set, val_set, train_idx, val_idx
        else:
            return train_set, val_set

    def normalize_tensor_2D(
            self,
            # At least of size (n, m)
            X,
    ):
        x_size = X.size()
        assert len(x_size) == 2, '2D tensor expected but got ' + str(len(x_size)) + 'D'
        X_nm = torch.clone(X)
        tmp = torch.pow(X_nm, exponent=2)
        sum_sqrt = torch.sqrt(torch.sum(tmp, dim=-1))
        # print('***** Sum sqrt')
        # print(sum_sqrt)
        for i in range(X_nm.size()[0]):
            X_nm[i] = X_nm[i] / max(sum_sqrt[i], 1e-9)
            # assert torch.absolute(torch.sum(X_nm[i] * X_nm[i]) - 1.0) < 0.000001
        return X_nm

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
        X_nm = torch.clone(X)
        if normalize_X:
            X_nm = self.normalize_tensor_2D(X=X)

        n_unique = len(torch.unique(labels))
        labels_unique = list(np.unique(labels))
        labels_unique.sort()
        list_0_to_n = list(range(n_unique))
        list_0_to_n.sort()
        assert labels_unique == list_0_to_n, 'Labels must be consecutive 0 to n, unique labels ' + str(labels_unique)
        self.logger.info('Fit mean with labels ' + str(labels) + ',X ' + str(X))

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
                self.logger.info(
                    'Loop label ' + str(i_lbl) + ', mean per label ' + str(mean_per_label) + ', x ' + str(x_tmp)
                    + ', centers ' + str(cluster_centers)
                )
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
        X_nm = torch.clone(X)
        if normalize_X:
            X_nm = self.normalize_tensor_2D(X=X)
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
        mapped_indexes_to_actual_labels = torch.floor(indexes_sorted / reps_per_label)

        # at this point all labels should appear the same number of times, exactly <reps_per_label> times

        # Remove duplicate labels in result if more than 1 mean_per_label
        # TODO what a mess this code, there should be a cleaner way
        if remove_dup_lbl:
            actual_labels_unique = torch.LongTensor()
            indexes_sorted_unique = torch.LongTensor()
            self.logger.debug(mapped_indexes_to_actual_labels)
            for i in range(mapped_indexes_to_actual_labels.size()[0]):
                row_actual_labels = mapped_indexes_to_actual_labels[i]
                row_indexes_sorted = indexes_sorted[i]
                j_positions = torch.LongTensor([0]*count_labels)
                for j in range(count_labels):
                    pos_j_all = (row_actual_labels == j).argwhere()
                    # all labels should appear <reps_per_label> times
                    assert len(pos_j_all) == reps_per_label, \
                        'row ' + str(i) + ' label ' + str(j) + ', inconsistency error'
                    # get first earliest position of j, by right should do torch.min()
                    pos_j = pos_j_all.flatten()[0]
                    j_positions[j] = pos_j
                    self.logger.debug('for row ' + str(row_actual_labels) + ' position of ' + str(j) + ': ' + str(pos_j))
                # sort from earliest positions to latest positions, so order is retained
                j_positions, _ = torch.sort(j_positions, descending=False, dim=0)
                tmp_actual_labels = row_actual_labels[j_positions]
                tmp_actual_labels = tmp_actual_labels.reshape(1, tmp_actual_labels.size()[0])
                tmp_indexes_sorted = row_indexes_sorted[j_positions]
                tmp_indexes_sorted = tmp_indexes_sorted.reshape(1, tmp_indexes_sorted.size()[0])
                # concatenate to final results
                actual_labels_unique = torch.cat((actual_labels_unique, tmp_actual_labels), dim=0)
                indexes_sorted_unique = torch.cat((indexes_sorted_unique, tmp_indexes_sorted), dim=0)
                # keep only those positions
                self.logger.debug('row ' + str(i) + ' j positions: ' + str(j_positions))
                self.logger.debug('row ' + str(i) + ' actual labels: ' + str(row_actual_labels) + ' --> ' + str(tmp_actual_labels))
                self.logger.debug('row ' + str(i) + ' index sorted: ' + str(row_indexes_sorted) + ' --> ' + str(tmp_indexes_sorted))
                self.logger.debug('row ' + str(i) + ' actual labels: ' + str(actual_labels_unique))
                self.logger.debug('row ' + str(i) + ' indexes sorted: ' + str(indexes_sorted_unique))
            indexes_final = indexes_sorted_unique
            actual_labels_final = actual_labels_unique
        else:
            indexes_final = indexes_sorted
            actual_labels_final = mapped_indexes_to_actual_labels

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

    # TODO Make as unit test below
    inputs = torch.Tensor([
        [0,1], [1,2], [2,3], [4,5], [5,6], [6,7], [7,8], [8,9], [9,10], [10,11],
    ])
    at_msk = torch.Tensor([True]*10)
    labels = torch.Tensor([0,0,0,0,0,1,1,1,2,2])
    fu = FitUtils()
    tset, vset = fu.training_data_split(
        inputs     = inputs,
        attn_masks = at_msk,
        labels     = labels,
        val_ratio  = 0.,
        stratify_label = True,
    )
    print('Training Set:')
    print(tset.tensors)
    print('Validation Set:')
    print(vset.tensors)

    exit(0)
