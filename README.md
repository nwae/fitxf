# fitxf

Library for fit transforms and searches.

```
pip install fitxf
```

## Basic Utilities

Cosine or dot similarity
```
from fitxf import TensorUtils
ts = TensorUtils()
x = np.random.rand(5,3)
y = np.random.rand(10,3)
ts.dot_sim(x,x)
ts.dot_sim(y,y)
ts.dot_sim(x,y)
```

## Fit Transform

