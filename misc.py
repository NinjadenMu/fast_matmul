import numpy as np
import time

#print(np.show_config())
rng = np.random.default_rng()

A = rng.random((4096, 4096))
B = rng.random((4096, 4096))

start = time.perf_counter()
C = A @ B
print(time.perf_counter() - start)