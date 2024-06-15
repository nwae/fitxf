# fitxf

Library for fit transforms and searches.

```
pip install fitxf
```

## Basic Utilities

Cosine or dot similarity
```
import numpy as np
from fitxf import TensorUtils
ts = TensorUtils()
x = np.random.rand(5,3)
y = np.random.rand(10,3)

# Cosine similarity
ts.similarity_cosine(x,x,'np')
ts.similarity_cosine(y,y,'np')
matches, dotsim = ts.similarity_cosine(x,y,'np')
print("matches",matches)
print("dot similarities",dotsim)

# Euclidean Distance
matches, dotsim = ts.similarity_distance(x,x,'np')
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

```
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

## Fit Transform

