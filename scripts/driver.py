#!/usr/bin/env python
import json
import matplotlib.pyplot as plt
import math
import subprocess
from tqdm import tqdm

BENCH_PATH = './bench'
PLOT_SAVE_PATH = 'plots/plot.png'
TIME_PER_IMPLEMENTATION = 5
IMPLEMENTATIONS = [
  'naive',
  'permuted',
  'tiled',
  'micro_kernel',
  'vectorized',
  'blis',
  'parallel'
]
IMPLEMENTATION_LABELS = {
  'naive': 'Naive',
  'permuted': 'Permuted',
  'tiled': 'Tiled',
  'micro_kernel': 'Micro-kernel',
  'vectorized': 'Vectorized Micro-kernel (ARM NEON)',
  'blis': 'Single-threaded BLIS-style',
  'parallel': 'Multi-threaded BLIS-style'
}


def benchmark(make_command, shell=False):
  line = {'x': [], 'y': []}
  n = 100
  time_elapsed = 0.0

  while time_elapsed < TIME_PER_IMPLEMENTATION:
    try:
      res = subprocess.run(
        make_command(n),
        capture_output=True,
        text=True,
        shell=shell,
        timeout=TIME_PER_IMPLEMENTATION - time_elapsed,
      )

      time_spent = json.loads(res.stdout)['avg_time']
      time_elapsed += float(time_spent[:-1])
      line['x'].append(time_elapsed)
      line['y'].append(n)

      #n += 10 ** int(math.log10(n))
      n += 100
    except Exception:
      break

  return line


implementation_lines = dict()

for implementation in tqdm(IMPLEMENTATIONS):
  implementation_lines[implementation] = benchmark(
    lambda n, impl=implementation: [BENCH_PATH, impl, str(n), '1']
  )

implementation_lines['Single-threaded OpenBLAS'] = benchmark(
  lambda n: f'OPENBLAS_NUM_THREADS=1 python scripts/openblas.py -n {n}',
  shell=True,
)

implementation_lines['Multi-threaded OpenBLAS'] = benchmark(
  lambda n: f'python scripts/openblas.py -n {n}',
  shell=True,
)

plt.figure(figsize=(10, 6))

for implementation, coords in implementation_lines.items():
  plt.plot(coords['x'], coords['y'], marker='o', 
           linestyle='-', label=IMPLEMENTATION_LABELS.get(implementation, implementation))

plt.xlabel('Time Elapsed (s)')
plt.ylabel('n')

plt.title('Largest Matrix Multiplied vs. Time Elapsed')
plt.legend(title='Implementations')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(PLOT_SAVE_PATH)
