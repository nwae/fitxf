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
ts.dot_sim(x,x,'np')
ts.dot_sim(y,y,'np')
matches, dotsim = ts.dot_sim(x,y,'np')
print("matches",matches)
print("dot similarities",dotsim)
```


## Fit Transform

