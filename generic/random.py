import time
import numpy as np

def numpy_seed(s=None):
    init_value = int(time.time()) if s is None else s
    np.random.seed(init_value)
