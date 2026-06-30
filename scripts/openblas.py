import argparse
import json
import numpy as np
import time

parser = argparse.ArgumentParser()

parser.add_argument('-n', type=int)

args = parser.parse_args()

A = np.random.rand(args.n, args.n)
B = np.random.rand(args.n, args.n)

start = time.perf_counter()
C = A @ B
end = time.perf_counter()

time_elapsed = end - start
gflops = 2 * args.n ** 3 / time_elapsed / 1e9

print(json.dumps({'avg_time': f'{time_elapsed}s', 'gflops': gflops}))
