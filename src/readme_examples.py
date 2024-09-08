## Basic Math

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


## Clustering

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


from fitxf import ClusterCosine
x = np.random.rand(20,3)
ClusterCosine().kmeans_optimal(x=x)


## Graph Wrappers

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


## Fit Transform

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


### Save Model to String & Load Back

# Save this json string somewhere
model_save = pca.model_to_b64json(numpy_to_base64_str=True, dump_to_b64json_str=True)

# Load back into new instance
new = FitXformPca()
new.load_model_from_b64json(model_b64json=model_save)
new.predict(X=x+np.random.rand(9,3))


### Fine Tuning

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
