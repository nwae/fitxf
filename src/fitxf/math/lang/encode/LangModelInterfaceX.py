import logging
import numpy as np
from fitxf.math.fit.transform.FitXformPca import FitXformPca


class LangModelInterfaceX:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def visualize_embedding(
            self,
            # numpy.ndarray type
            encoding_np,
            # anything to label the encoding
            labels_list,
    ):
        pca = FitXformPca(logger=self.logger)
        pca.create_scatter_plot2d(
            x_transform = pca.fit(X=encoding_np, n_components=2),
            labels_list = labels_list,
            show = True,
            save_filepath = None,
        )
        return


if __name__ == '__main__':
    # Fake data
    x = np.array([
        [3, 16, 20], [2, 18, 21], [4, 17, 19],
        [15, 19, 5], [17, 22, 5], [16, 20, 3],
    ])
    labels = ['dog', 'dog', 'dog', 'cat', 'cat', 'cat']
    obj = LangModelInterfaceX()
    obj.visualize_embedding(encoding_np=x, labels_list=labels)
    exit(0)
