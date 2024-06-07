import logging
import torch
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from nwae.math.fit.utils.TensorUtils import TensorUtils
from nwae.math.fit.cluster.Cluster import Cluster
from nwae.math.utils import Logging


class FitVoronoi:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.cluster = Cluster(logger=self.logger)
        self.tensor_utils = TensorUtils(logger=self.logger)
        return

    def __convert_tensor_to_numpy(
            self,
            x,
    ):
        if type(x) is torch.Tensor:
            return x.cpu().detach().numpy()
        assert type(x) is np.ndarray
        return x

    def __convert_tensor_to_torch(self, x):
        if type(x) is np.ndarray:
            return torch.from_numpy(x)
        assert type(x) is torch.Tensor
        return x

    def fit_voronoi(
            self,
            points,
    ):
        # TODO group into bigger regions
        points = self.__convert_tensor_to_numpy(x=points)
        vor = Voronoi(points)

        x_arg = np.argsort(a=points[:,0])
        points_ordered = points[x_arg]
        self.logger.info('Ordered points: ' + str(points_ordered))

        voronoi = {}
        for i, pnt in enumerate(points_ordered):
            voronoi[i] = {'point': pnt}
            # TODO Find distance to all points <i, then by nearest to furthest
            prev_points = points_ordered[0:i]
            self.logger.debug('Prev points ' + str(i) + ': ' + str(prev_points))
        return vor

    def predict(
            self,
            x,
            x_ref,
    ):
        x_np = self.__convert_tensor_to_numpy(x=x)
        x_ref_np = self.__convert_tensor_to_numpy(x=x_ref)
        top_labels, probs = self.tensor_utils.similarity_distance(
            x = x_np,
            ref = x_ref_np,
            return_tensors = 'np',
        )
        return top_labels, probs


class FitVoronoiUnitTest:
    def __init__(self, logger):
        self.logger = logger
        return

    def test(
            self,
    ):
        fg = FitVoronoi(logger=self.logger)
        test_points_voronoi = torch.FloatTensor([
            [-0.2, 1.],
            [1., 0.], [1.2, 2.],
            [2., -0.6], [2.3, 5.],
        ])
        res = fg.fit_voronoi(
            points = test_points_voronoi,
        )
        print('vertices', res.vertices)
        print('ridge vertices', res.ridge_vertices)
        print('regions', res.regions)
        voronoi_plot_2d(res)
        plt.show()
        print('ALL TESTS PASSED OK')
        return


if __name__ == '__main__':
    FitVoronoiUnitTest(
        logger = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    ).test()
    exit(0)
