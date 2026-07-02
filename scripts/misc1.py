import subprocess
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

data = []
for i in tqdm(range(81)):
  faster = float(json.loads(subprocess.run(['./bench', 'blis', str(i * 32 + 768), '5'], 
                  capture_output=True, text=True).stdout)['gflops'])
  
  slower = float(json.loads(subprocess.run(['./bench1', 'blis', str(i * 32 + 768), '5'], 
                  capture_output=True, text=True).stdout)['gflops'])
  
  data.append((i * 32 + 768, slower / faster))

# Unpack the tuples into separate x and y lists
x, y = zip(*data)

plt.plot(x, y)
plt.xlabel('n')
plt.ylabel('nc=16 GFLOPs / nc=612 GFLOPS')
plt.show()