#!/usr/bin/env -S OPENBLAS_NUM_THREADS=1 python
import numpy as np
import time
import os
import copy
import json
import random
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt

A = np.random.rand(8192, 8192)
B = np.random.rand(8192, 8192)

start = time.perf_counter()
#C = A @ B
print(f'wall clock: {time.perf_counter() - start}')

BENCH_PATH = './bench'
TARGET = 'parallel'
TRIALS = 1
TEST_SIZES = [4100]

# Parameters to search over and (min, max, min_step) defining search space axes
SEARCH_SPACE = {
  'mc': (12, 2040, 12), 
  'nc': (8, 2048, 8), 
  'kc': (16, 2048, 8)
}
MAX_POINTS = 300

# Global list to track evaluations for visualization
EVALUATION_HISTORY = []

def get_score(env):
  total_gflops = 0
  for test_size in TEST_SIZES:
    res = subprocess.run([BENCH_PATH, TARGET, str(test_size), str(TRIALS)], 
                          capture_output=True, text=True, env=env)
    
    if len(res.stderr.strip()) != 0:
      continue
    
    # Handle potential JSON parsing errors safely
    try:
      total_gflops += float(json.loads(res.stdout)['gflops'])
    except (json.JSONDecodeError, KeyError):
      continue

  # Record the configuration and its score for plotting
  if total_gflops > 0:
      try:
          mc = int(env.get('mc', 0))
          nc = int(env.get('nc', 0))
          kc = int(env.get('kc', 0))
          EVALUATION_HISTORY.append({'mc': mc, 'nc': nc, 'kc': kc, 'gflops': total_gflops})
      except ValueError:
          pass

  return total_gflops
 
def best_in_space(curr_config, parameters_to_search: list[str], search_space=SEARCH_SPACE):
  if len(parameters_to_search) == 0:
    env = os.environ.copy()
    for parameter, val in curr_config.items():
      env[parameter] = str(val)

    return dict(), get_score(env)
  
  else:
    parameter = parameters_to_search[0]
    parameter_min = search_space[parameter][0]
    parameter_max = search_space[parameter][1] + 1
    parameter_step = search_space[parameter][2]

    best_score = 0
    best_config = dict() 
    
    for val in range(parameter_min, parameter_max, parameter_step):
      curr_config[parameter] = val

      config, score = best_in_space(curr_config, parameters_to_search[1:], search_space)
      if score >= best_score: 
        best_config = config
        best_config[parameter] = val
        best_score = score

    return best_config, best_score

def grid_search(max_points=MAX_POINTS):
  if max_points == None:
    return best_in_space(dict(), list(SEARCH_SPACE.keys()))
  
  def num_points_in_space(search_space):
    num_points = 1
    for span in search_space.values():
      num_points *= (span[1] - span[0]) // span[2] + 1
    return num_points
  
  assert(max_points >= 2 ** len(SEARCH_SPACE.keys()))

  search_space = copy.copy(SEARCH_SPACE)
  while num_points_in_space(search_space) > max_points:
    prev_points = num_points_in_space(search_space)
    for parameter in search_space.keys():
      parameter_min = search_space[parameter][0]
      parameter_max = search_space[parameter][1]
      parameter_step = min(search_space[parameter][2] * 2, parameter_max - parameter_min)
    
      search_space[parameter] = (parameter_min, parameter_max, parameter_step)
      
    if num_points_in_space(search_space) == prev_points:
        print("Warning: Max points constraint cannot be met. Exiting space reduction.")
        break
      
  return best_in_space(dict(), list(search_space.keys()), search_space)

def random_search(max_points=MAX_POINTS):
  env = os.environ.copy()
  best_config = dict()
  best_score = 0
  for i in tqdm(range(max_points), desc="Random Search"):
    config = dict()
    for parameter, span in SEARCH_SPACE.items():
      val = random.randrange(span[0], span[1] + 1, span[2])
      env[parameter] = str(val)
      config[parameter] = val

    score = get_score(env)
    if score > best_score:
      best_config = config
      best_score = score

    if i % 50 == 0:
      print(best_config, best_score)

  return best_config, best_score

def genetic_algorithm(max_points=MAX_POINTS, pop_size=20, mutation_rate=0.3):
  """
  Optimizes BLIS parameters using a Genetic Algorithm with Tournament Selection,
  Uniform Crossover, and Elitism.
  """
  env = os.environ.copy()
  best_config = dict()
  best_score = 0
  memo = {} # Cache to prevent re-evaluating the same genes
  eval_count = 0

  def generate_random_individual():
    return {param: random.randrange(span[0], span[1] + 1, span[2]) 
            for param, span in SEARCH_SPACE.items()}

  def evaluate(ind):
    nonlocal eval_count, best_score, best_config
    # Use a sorted tuple of items as a dictionary key for caching
    key = tuple(sorted(ind.items()))
    if key in memo:
      return memo[key]
    
    if eval_count >= max_points:
      return 0
        
    for param, val in ind.items():
      env[param] = str(val)
        
    score = get_score(env)
    memo[key] = score
    eval_count += 1
    
    if score > best_score:
      best_score = score
      best_config = ind.copy()
      tqdm.write(f"Eval {eval_count:03d} | New Best: {best_config} | Score: {best_score:.2f} GFLOPS")
        
    return score

  # 1. Initialize Population
  # Bound pop_size to max_points in case max_points is unusually small
  population = [generate_random_individual() for _ in range(min(pop_size, max_points))]
  
  with tqdm(total=max_points, desc="GA Evaluations") as pbar:
    # Evaluate initial population
    fitnesses = []
    for ind in population:
      if eval_count >= max_points: break
      prev_eval = eval_count
      fit = evaluate(ind)
      fitnesses.append(fit)
      if eval_count > prev_eval: pbar.update(1)

    # 2. Evolution Loop
    while eval_count < max_points:
      new_population = []
      
      # Elitism: Carry over the top performer
      if fitnesses:
        best_idx = fitnesses.index(max(fitnesses))
        new_population.append(population[best_idx])
      
      while len(new_population) < pop_size and eval_count < max_points:
        # Tournament Selection
        participants1 = random.sample(list(zip(population, fitnesses)), min(3, len(population)))
        parent1 = max(participants1, key=lambda x: x[1])[0]
        
        participants2 = random.sample(list(zip(population, fitnesses)), min(3, len(population)))
        parent2 = max(participants2, key=lambda x: x[1])[0]
        
        # Uniform Crossover
        child = {p: parent1[p] if random.random() < 0.5 else parent2[p] 
                 for p in SEARCH_SPACE.keys()}
        
        # Mutation
        if random.random() < mutation_rate:
          param_to_mutate = random.choice(list(SEARCH_SPACE.keys()))
          span = SEARCH_SPACE[param_to_mutate]
          # Pick a new random value on the valid grid
          child[param_to_mutate] = random.randrange(span[0], span[1] + 1, span[2])
            
        new_population.append(child)
          
      # Assign the new generation and evaluate
      population = new_population
      fitnesses = []
      for ind in population:
        if eval_count >= max_points: break
        prev_eval = eval_count
        fit = evaluate(ind)
        fitnesses.append(fit)
        if eval_count > prev_eval: pbar.update(1)

  return best_config, best_score

def plot_evaluations():
  if not EVALUATION_HISTORY:
    print("No valid benchmark data gathered to plot.")
    return

  # Extract parameters for plotting
  mcs = [h['mc'] for h in EVALUATION_HISTORY]
  ncs = [h['nc'] for h in EVALUATION_HISTORY]
  kcs = [h['kc'] for h in EVALUATION_HISTORY]
  gflops = [h['gflops'] for h in EVALUATION_HISTORY]

  # Create a 2x2 grid of subplots
  fig = plt.figure(figsize=(16, 12))

  # 1. 3D Plot showing interactions
  ax1 = fig.add_subplot(2, 2, 1, projection='3d')
  sc1 = ax1.scatter(mcs, ncs, kcs, c=gflops, cmap='viridis', s=50, alpha=0.8)
  ax1.set_xlabel('mc')
  ax1.set_ylabel('nc')
  ax1.set_zlabel('kc')
  ax1.set_title('3D Parameter Space')
  fig.colorbar(sc1, ax=ax1, label='Total GFLOPS')

  # 2. 2D Plot: mc vs GFLOPS
  ax2 = fig.add_subplot(2, 2, 2)
  ax2.scatter(mcs, gflops, c=gflops, cmap='viridis', alpha=0.7)
  ax2.set_xlabel('mc')
  ax2.set_ylabel('GFLOPS')
  ax2.set_title('mc vs GFLOPS')

  # 3. 2D Plot: nc vs GFLOPS
  ax3 = fig.add_subplot(2, 2, 3)
  ax3.scatter(ncs, gflops, c=gflops, cmap='viridis', alpha=0.7)
  ax3.set_xlabel('nc')
  ax3.set_ylabel('GFLOPS')
  ax3.set_title('nc vs GFLOPS')

  # 4. 2D Plot: kc vs GFLOPS
  ax4 = fig.add_subplot(2, 2, 4)
  ax4.scatter(kcs, gflops, c=gflops, cmap='viridis', alpha=0.7)
  ax4.set_xlabel('kc')
  ax4.set_ylabel('GFLOPS')
  ax4.set_title('kc vs GFLOPS')

  plt.tight_layout()
  plt.savefig('autotuner_visualization.png')
  print("Visualization saved to autotuner_visualization.png")

if __name__ == '__main__':
  # Replaced random_search with genetic_algorithm
  print("Starting Genetic Algorithm Optimizer...")
  best_config, best_score = random_search(500)
  
  print(f"\nOptimization Complete.")
  print(f"Best Config: {best_config}")
  print(f"Best Score: {best_score:.2f} GFLOPS")
  
  # Trigger the plot rendering
  plot_evaluations()