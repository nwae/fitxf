import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fitxf.math.fit.utils.FitUtils import FitUtils
from sklearn.decomposition import PCA
from fitxf.math.fit.utils.TensorUtils import TensorUtils
from fitxf.math.utils.EnvironRepo import EnvRepo
from fitxf.math.utils.Logging import Logging


#
# A general interface to transform original points X with associated labels into a transform X_ = f(X)
# X_ is possibly a compressed version, but not necessarily.
# If X_ is a clustered form, this means points in the same cluster will have the exact same values,
# in the usual case the center of the cluster value.
# The entire purpose of Fit transformation is mainly to compress data, to economize RAM.
# Examples:
#    Given
#       X = [[1,2,3], [3,2,1], [-1,-2,-2], [-3,-4,-2]]
#       X_labels = ['+', '+', '-', '-']
#    and arbitrary X_any = [[9,9,8], [-55,-33,-55]]
#    PCA Transform 1-dim:
#       [[ 3.72936414], [ 3.81640291], [-2.55648061], [-4.98928643]]
#       predict X_any: [['+', '+', '-', '-'], ['-', '-', '+', '+']]
#    Cluster Transform:
#
class FitXformInterface:

    KEY_X_TRANSFORM = 'X_transform'
    KEY_X_TRANSFORM_CHECK = 'X_transform_check'
    KEY_X_LABELS = 'X_labels'
    KEY_X_FULL_RECS = 'X_full_records'
    # Inverse PCA/Cluster transform
    KEY_X_INV_TRANSFORM = 'X_inverse_transform'
    KEY_CENTROID = 'centroid'
    KEY_N_COMPONENTS_OR_CENTERS = 'n'
    KEY_PRINCIPAL_COMPONENTS = 'principal_components'
    KEY_CENTERS = 'centers'
    KEY_GRID_VECTORS = 'grid_vectors'
    KEY_GRID_NUMBERS = 'grid_numbers'

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.tensor_utils = TensorUtils(logger=self.logger)
        self.X_full_records = None

        # Data/labels
        self.X = None
        self.X_labels = None
        self.X_full_records = None
        self.X_transform = None
        self.X_transform_check = None
        # Inverse PCA transform
        self.X_inverse_transform = None
        # Create an artificial grid
        self.X_grid_vectors = None
        self.X_grid_numbers = None
        # Measures for us to optimize the number of optimal number of pca components
        self.grid_density = None
        self.grid_density_mean = None
        # same as cluter inertia or if not cluster, is error of distances between point and transform summed up
        self.distance_error = None
        self.distance_error_mean = None
        self.angle_error = None
        self.angle_error_mean = None
        return

    def is_model_ready(self):
        raise Exception('Must be implemented by derived class')

    # Will dynamically look for optimal model parameters, on the condition of the target grid density
    def fit_optimal(
            self,
            X: np.ndarray,
            X_labels = None,
            X_full_records = None,
            # may not apply to all models. for clusters, grid is just the cluster.
            # for PCA, we need to predefine how to calculate the grid.
            target_grid_density = 2, # at least 2 points per grid
            # allowed values 'median', 'mean', 'min'
            measure = 'median',
            # Model dependent interpretation, or ignore if not relevant for specific model
            min_components = 1,
            max_components = 99999,
    ) -> dict:
        raise Exception('Must be implemented by derived class')

    def fit(
            self,
            X: np.ndarray,
            X_labels = None,
            X_full_records = None,
            # Model dependent interpretation, or ignore if not relevant for specific model
            # For example, can mean how many clusters, or how many PCA components, or how many to sample
            # in a discrete Fourier transform, etc.
            n_components = 2,
            # for fine-tuning already trained clusters, thus no need to start all over again
            # useful for clusters of more than 1,000,000 points for example, where starting
            # again means another half day of fit training
            start_centers: np.ndarray = None,
    ) -> dict:
        raise Exception('Must be implemented by derived class')

    # Recover estimate of original point from PCA compression
    def inverse_transform(
            self,
            x_transform: np.ndarray,
    ):
        raise Exception('Must be implemented by derived class')

    # Get PCA values of arbitrary points
    def calc_transform(
            self,
            X: np.ndarray,
    ):
        raise Exception('Must be implemented by derived class')

    def predict(
            self,
            X: np.ndarray,
            top_k = 5,
            return_full_record = False,
            use_grid = False,
    ):
        raise Exception('Must be implemented by derived class')

    def predict_standard(
            self,
            X: np.ndarray,
            ref_X: np.ndarray,
            ref_labels: np.ndarray,
            ref_full_records: list = (),
            top_k = 5,
            similarity_type = 'cosine',
            return_full_record = False,
    ):
        """
        result_ordered returns 2D matrix, each row represents the closest indexes of ref ordered
        So means if the row is [5, 0, 1, 3, 2, 4], means that the closest match is index 5 in ref,
        2nd closest is index 0 in ref, and so on.
        m_dot_ordered is the result of the dot product, a value of magnitude equal or less than 1.
        """
        self.logger.info('For prediction, using similarity type "' + str(similarity_type) + '"')
        if similarity_type == 'distance':
            result_ordered, m_dot_ordered = self.tensor_utils.similarity_distance(
                x = X,
                ref = ref_X,
                return_tensors = 'np',
            )
        else:
            result_ordered, m_dot_ordered = self.tensor_utils.dot_sim(
                x = X,
                ref = ref_X,
                return_tensors = 'np',
            )

        self.logger.debug(
            'Result ordered for top k ' + str(top_k) + ': ' + str(result_ordered) + ', ref length ' + str(len(ref_X))
        )

        if return_full_record:
            assert ref_full_records is not None, 'Cannot return full records'
            pred_records = [[ref_full_records[i] for i in np_row] for np_row in result_ordered]
            pred_probs_list = m_dot_ordered.tolist()

            return [ar[0:min(top_k, len(ar))] for ar in pred_records], \
                [ar[0:min(top_k, len(ar))] for ar in pred_probs_list]
        else:
            pred_labels = ref_labels[result_ordered].tolist()
            pred_probs_list = m_dot_ordered.tolist()

            return [ar[0:min(top_k, len(ar))] for ar in pred_labels], \
                [ar[0:min(top_k, len(ar))] for ar in pred_probs_list]

    def create_scatter_plot2d(
            self,
            x_transform: np.ndarray,
            labels_list,
            show = False,
            save_filepath = None,
            add_noise = False,
    ):
        if x_transform.shape[-1] != 2:
            self.logger.info('X not 2D, with shape ' + str(x_transform.shape) + '. Using PCA to reduce to 2D..')
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(pd.DataFrame(x_transform))
        else:
            X_2d = np.array(x_transform)

        if add_noise:
            fu = FitUtils(logger=self.logger)
            ds = fu.get_point_distances(np_tensors=X_2d)
            dist_median = np.median(ds)
            # 1/5 of average point distances on average
            X_2d += np.random.random(size=X_2d.shape) * dist_median / 2.5

        columns = ['PC'+str(i+1) for i in range(2)]
        # Create a new DataFrame with reduced dimensions
        df_reduced = pd.DataFrame(X_2d, columns=columns)

        plt.scatter(df_reduced['PC1'], df_reduced['PC2'], label=labels_list)

        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Scatter Plot')

        # Add labels to each dot
        # for i, label in enumerate(embeddings_df.iloc[:, -1].to_list()):
        for i, label in enumerate(labels_list):
            plt.text(df_reduced['PC1'][i], df_reduced['PC2'][i], label)

        if show:
            plt.show()

        # Save and display the plot
        if save_filepath is not None:
            plt.savefig(save_filepath, format='png')
        return plt


class SampleFit(FitXformInterface):

    def __init__(
            self,
            logger = None,
    ):
        super().__init__(logger=logger)
        self.model_centroid = None
        self.pca_components_2d = None
        return

    def fit(
            self,
            X: np.ndarray,
            X_labels = None,
            X_full_records = None,
            # Model dependent interpretation, or ignore if not relevant for specific model
            # For example, can mean how many clusters, or how many PCA components, or how many to sample
            # in a discrete Fourier transform, etc.
            n_components = 2,
            # for fine-tuning already trained clusters, thus no need to start all over again
            # useful for clusters of more than 1,000,000 points for example, where starting
            # again means another half day of fit training
            start_centers: np.ndarray = None,
            return_details = False,
    ):
        pca = PCA(n_components=n_components)

        # Create copy, don't use same ref from input X
        self.X = np.array(X)
        self.X_transform = pca.fit_transform(pd.DataFrame(self.X))
        self.X_labels = np.array(X_labels)
        self.X_full_records = X_full_records
        self.principal_components = pca.components_
        self.centroid = np.mean(X, axis=0)
        return self.X_transform


class UnitTest:

    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self, plot_chart):
        from fitxf.math.lang.encode.LangModelPt import LangModelPt as LmPt
        ft = SampleFit(logger=self.logger)

        texts = [
            "Let's have coffee", "Free for a drink?", "How about Starbucks?",
            "I am busy", "Go away", "Don't disturb me",
            "Monetary policies", "Interest rates", "Deposit rates",
        ]
        labels = ['d', 'd', 'd', 'x', 'x', 'x', '$', '$', '$']
        lmo = LmPt(lang='en', cache_folder=EnvRepo(repo_dir=os.environ.get("REPO_DIR", None)).MODELS_PRETRAINED_DIR)

        embeddings = lmo.encode(text_list=texts, return_tensors='np')

        x_compressed = ft.fit(X=embeddings)

        small_random_values = (np.random.random(size=x_compressed.shape) - 0.5) / 100
        pred_labels, pred_probs = ft.predict_standard(
            X = x_compressed + small_random_values,
            ref_X = x_compressed,
            ref_labels = np.array(labels),
        )
        prd_labels_top = [r[0] for r in pred_labels]
        assert prd_labels_top == labels, 'Predicted top labels not expected ' + str(zip(prd_labels_top, labels))
        prd_probs_top = [r[0] for r in pred_probs]
        assert np.min(np.array(prd_probs_top)) > 0.999, 'Top probabilities ' + str(prd_probs_top)

        if plot_chart:
            ft.create_scatter_plot2d(
                x_transform = embeddings,
                labels_list = zip(labels, texts),
                show = True,
            )
        print('ALL TESTS PASSED OK')
        return


if __name__ == '__main__':
    UnitTest(logger=Logging.get_default_logger(log_level=logging.INFO, propagate=False)).test(
        plot_chart = True,
    )
    exit(0)
