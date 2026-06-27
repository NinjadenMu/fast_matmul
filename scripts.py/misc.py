#!/usr/bin/env -S OPENBLAS_NUM_THREADS=1 python
import numpy as np
import time
import os

A = np.random.rand(8192, 8192)
B = np.random.rand(8192, 8192)

start = time.perf_counter()
C = A @ B
print(f'wall clock: {time.perf_counter() - start}')