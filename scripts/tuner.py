"""
A basic autotuner that can tune an arbitrary number of parameters to maximize 
gflops. Grid search and a random tuner are implemented. 

The random tuner works better for an equivalent amount of compute, 
but it seems like the benefits of autotuning vs. choosing some reasonable 
values by hand are fairly minimal.
"""

import copy
import json
import os
import random
import subprocess
import typing
from tqdm import tqdm

BENCH_PATH = './bench'
TARGET = 'blis'
TRIALS = 1
TEST_SIZES = [500, 1000, 1500, 2000, 3000]
# Parameters to search over and (min, max, min_step) defining search space axes
SEARCH_SPACE = {
  'mc': (12, 2040, 12), 
  'nc': (8, 2048, 8), 
  'kc': (16, 2048, 8)
}
MAX_POINTS = 300

def get_score(env):
  total_gflops = 0
  for test_size in TEST_SIZES:
    res = subprocess.run([BENCH_PATH, TARGET, str(test_size), str(TRIALS)], 
                          capture_output=True, text=True, env=env)
    
    if len(res.stderr.strip()) != 0:
      continue

    total_gflops += float(json.loads(res.stdout)['gflops'])

  return total_gflops / len(TEST_SIZES)
 
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

    best_config = dict()
    best_score = 0
    for val in range(parameter_min, parameter_max, parameter_step):
      curr_config[parameter] = val

      config, score = best_in_space(curr_config, parameters_to_search[1:], search_space)
      if score > best_score:
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
    for parameter in search_space.keys():
      parameter_min = search_space[parameter][0]
      parameter_max = search_space[parameter][1]
      parameter_step = min(search_space[parameter][2] * 2, parameter_max - parameter_min)
    
      search_space[parameter] = (parameter_min, parameter_max, parameter_step)
      
  return best_in_space(dict(), list(search_space.keys()), search_space)

def random_search(max_points=MAX_POINTS):
  env = os.environ.copy()
  best_config = dict()
  best_score = 0
  for i in tqdm(range(max_points)):
    config = dict()
    for parameter, span in SEARCH_SPACE.items():
      val = random.randrange(span[0], span[1] + 1, span[2])
      env[parameter] = str(val)
      config[parameter] = val

    score = get_score(env)
    if score > best_score:
      best_config = config
      best_score = score

  return best_config, best_score

if __name__ == '__main__':
  print(random_search(300))
