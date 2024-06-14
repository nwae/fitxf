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

## Fit Transform

