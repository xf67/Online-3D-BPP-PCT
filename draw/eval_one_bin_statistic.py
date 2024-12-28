import re
import matplotlib.pyplot as plt
import numpy as np

with open('logger/mcts3-500.txt', 'r') as f:
    content = f.read()

eps_ratios = re.findall(r'Episode ratio: ([0-9.]+)', content)
eps_ratios = [float(x) for x in eps_ratios]

bin_width = 0.025 
bins = np.arange(0.0, 0.8, bin_width)
plt.figure(figsize=(10, 6))
plt.hist(eps_ratios, bins=bins, edgecolor='black')
plt.title('Distribution of Episode Ratio')
plt.xlabel('Episode Ratio Value')
plt.ylabel('Frequency')

plt.xlim(0.1, 0.8)
plt.ylim(0, 20) 
plt.yticks(np.arange(0, 20, 2.5))

plt.axvline(np.mean(eps_ratios), color='red', linestyle='dashed', linewidth=1, 
            label=f'Mean: {np.mean(eps_ratios):.3f}')
plt.axvline(np.median(eps_ratios), color='green', linestyle='dashed', linewidth=1, 
            label=f'Median: {np.median(eps_ratios):.3f}')

print(f'Mean: {np.mean(eps_ratios):.4f}')
print(f'Median: {np.median(eps_ratios):.4f}')
print(f'Std: {np.std(eps_ratios):.4f}')
print(f'Min: {min(eps_ratios):.4f}')
print(f'Max: {max(eps_ratios):.4f}')

plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('draw/eps_ratio_distribution.png')
plt.close()
