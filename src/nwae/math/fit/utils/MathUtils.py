import numpy as np


class MathUtils:
  
    def __init__(self):
      return
  
    def sample_random_no_repeat(
            self,
            list,
            n,
    ):
        assert n <= len(list)
        rng = np.random.default_rng()
        numbers = rng.choice(len(list), size=n, replace=False)
        sampled = []
        for i in numbers:
            sampled.append(list[i])
        return sampled
