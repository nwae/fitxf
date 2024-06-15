# fitxf

Just a simple math utility library

- Basic math that don't exist in numpy as a single function call
- Simple math for optimal clusters that can't seem to find in numpy
  or sklearn
- Simple wrappers to simplify tensor transforms or compression
  via clustering or PCA, and allow to do searches using transformed
  data

```
pip install fitxf
```

## Basic Math

Cosine or dot similarity between multi-dim vectors
```
import numpy as np
from fitxf import TensorUtils
ts = TensorUtils()
x = np.random.rand(5,3)
y = np.random.rand(10,3)

# Cosine similarity, find closest matches of each vector in x
# with all vectors in ref
# For Euclidean distance, just replace with "similarity_distance"
ts.similarity_cosine(x=x, ref=x, return_tensors='np')
ts.similarity_cosine(x=y, ref=y, return_tensors='np')
matches, dotsim = ts.similarity_cosine(x=x, ref=y, return_tensors='np')
print("matches",matches)
print("dot similarities",dotsim)
```

## Clustering

Auto clustering into optimal n clusters, via heuristic manner

### Case 1: All clusters (think towns) are almost-equally spaced

- in this case, suppose optimal cluster centers=n (think
  salesmen)
- if number of clusters k<n, then each salesman need to cover
  a larger area, and their average distances from each other is smaller
- if number of clusters k>n, then things become a bit crowded,
  with more than 1 salesman covering a single town
- Thus at transition from n --> n+1 clusters, the average
  distance between cluster centers will decrease

### Case 2: Some clusters are spaced much larger apart

In this case, there will be multiple turning points, and we
may take an earlier turning point or later turning points

Optimal cluster by Euclidean Distance
```
from fitxf import Cluster
x = np.array([
    [5, 1, 1], [8, 2, 1], [6, 0, 2],
    [1, 5, 1], [2, 7, 1], [0, 6, 2],
    [1, 1, 5], [2, 1, 8], [0, 2, 6],
])
obj = Cluster()
obj.kmeans_optimal(
    x = x,
    estimate_min_max = True,
)
```

Optimal cluster by cosine distance
```
from fitxf import ClusterCosine
x = np.random.rand(20,3)
ClusterCosine().kmeans_optimal(x=x)
```

## Fit Transform

Convenient wrapper
- fit a set of vectors into compressed PCA, clusters, etc.
- predict via cosine similarity, Euclidean distance of arbitrary
  vectors
- fine tune


Sample code for basic training to transform data -
```
from fitxf import FitXformPca, FitXformCluster
import numpy as np
x = np.array([
    [5, 1, 1], [8, 2, 1], [6, 0, 2],
    [1, 5, 1], [2, 7, 1], [0, 6, 2],
    [1, 1, 5], [2, 1, 8], [0, 2, 6],
])
user_labels = [
    'a', 'a', 'a',
    'b', 'b', 'b',
    'c', 'c', 'c',
]
pca = FitXformPca()
res_fit_pca = pca.fit_optimal(X=x, X_labels=user_labels)
print('X now reduced to\n',res_fit_pca['X_transform'])

cls = FitXformCluster()
res_fit_cls = cls.fit_optimal(X=x, X_labels=user_labels)
print('X now reduced to\n',res_fit_cls['X_transform'])

pca.predict(X=x+np.random.rand(9,3))
cls.predict(X=x+np.random.rand(9,3))
```

Sample code to save and load model -
```
import json
# Save this json string somewhere
model_save = pca.model_to_json(numpy_to_base64_str=True, dump_to_json_str=True)

# Load back into new instance
new = FitXformPca()
new.load_model_from_json(model_json=json.loads(model_save))
new.predict(X=x+np.random.rand(9,3))
```