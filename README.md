# fitxf

Simple math utility library

- Basic math that don't exist in numpy as single function call
- Simple math for optimal clusters not in sklearn
- Simple graph wrappers
- Simple transform wrappers to simplify tensor transforms or
  compression via clustering (Euclid distance or cosine) or PCA
- allow to do searches using transformed data
- allow to save/load transform model to/from string & fine-tuning

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

## Graph Wrappers

Simple directed graph with Dijkstra or simple path.

```
from fitxf import GraphUtils
gu = GraphUtils()
G = gu.create_multi_graph(
    edges = [
        {'key': 'plane', 'u': 'Shanghai', 'v': 'Tokyo', 'distance': 10},
        {'key': 'ship', 'u': 'Shanghai', 'v': 'Tokyo', 'distance': 100},
        {'key': 'plane', 'u': 'Tokyo', 'v': 'Shanghai', 'distance': 22},
        {'key': 'plane', 'u': 'Tokyo', 'v': 'Seoul', 'distance': 5},
        {'key': 'plane', 'u': 'Seoul', 'v': 'Tokyo', 'distance': 6},
        {'key': 'ship', 'u': 'Seoul', 'v': 'Tokyo', 'distance': 60},
    ],
    col_weight = 'distance',
    directed = True,
)
# Shanghai-->Tokyo-->Seoul, total weight 15
print(gu.get_paths(G=G, source="Shanghai", target="Seoul", method="dijkstra"))
# Shanghai-->Tokyo-->Seoul, total weight 105
print(gu.get_paths(G=G, source="Shanghai", target="Seoul", method="simple", agg_weight_by="max"))
# Seoul-->Tokyo-->Shanghai, total weight 28
print(gu.get_paths(G=G, source="Seoul", target="Shanghai", method="dijkstra"))
# Shanghai-->Tokyo-->Seoul, total weight 82
print(gu.get_paths(G=G, source="Seoul", target="Shanghai", method="simple", agg_weight_by="max"))
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

From above code same you will see the original X of 3 dimensions
were reduced to 2.
```
X now reduced to
 [[ 3.31282121 -0.0595825 ]
 [ 5.50813438  0.41970015]
 [ 4.04773021 -1.54803569]
 [-1.35557188  3.11843837]
 [-1.16795985  4.47103187]
 [-2.95485944  3.21899561]
 [-1.77834736 -2.52015183]
 [-2.23431234 -4.68080139]
 [-3.37763491 -2.41959458]]
```

whereas for clustering
```
X now reduced to
 [2 2 2 0 0 0 1 1 1]
```
where each point in 3 dimensions is represented by only a scalar
center label.


### Save Model to String & Load Back

Sample code to save and load model -
```
# Save this Base64 string somewhere
model_save = pca.model_to_b64json(numpy_to_base64_str=True, dump_to_b64json_str=True)

# Load back into new instance
new = FitXformPca()
new.load_model_from_b64json(model_b64json=model_save)
new.predict(X=x+np.random.rand(9,3))
```

### Fine Tuning

After saving models & loading back, one can fine tune with new
points.

Test fine tune with same centers and data

```
cls = FitXformCluster()
res_fit_cls = cls.fit_optimal(X=x, X_labels=user_labels)
centers = res_fit_cls["centers"]

res = cls.fine_tune(
   X = x,
   X_labels = user_labels,
   n_components = 3,
)
[print(k,v) for k,v in res.items()]
print('Expect 1 iteration, got ', res["n_iter"])

print('old centers',centers)
print('new centers',res['centers'])
```

Fine tune with new points

```
x = np.array([
    [5, 1, 1], [8, 2, 1], [6, 0, 2],
    [1, 5, 1], [2, 7, 1], [0, 6, 2],
    # remove last 2 points
    [1, 1, 5], # [2, 1, 8], [0, 2, 6],
    # new points
    [4, 2, 1], [0, 6, 2], [1, 1, 7],
])
user_labels = [
    'a', 'a', 'a',
    'b', 'b', 'b',
    # remove last 2 points
    'c', # 'c', 'c',
    'a', 'b', 'c',
]
res_fit_new = cls.fine_tune(X=x, X_labels=user_labels, n_components=3)

print('old centers',centers)
print('new centers',res_fit_new['centers'])
```

### Tensor DB Models

The math is simple, but the technical implementation is orders
of magnitude far more complicated.
Transformed data needs to be updated to storage for subsequent
level searches, model & data needs to be kept in sync, etc.

To be updated..


### Miscellaneous

