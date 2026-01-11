import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = os.path.join('logs','convergence','AAPL','SAC_seed42','episode_rewards.csv')
vis_dir = 'visualizations'
os.makedirs(vis_dir, exist_ok=True)

episodes = []
rewards = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        episodes.append(int(row['episode']))
        rewards.append(float(row['reward']))

rewards = np.array(rewards)
# moving average
window = 20
ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
ma_x = episodes[window-1:]

plt.figure(figsize=(10,5))
plt.plot(episodes, rewards, color='lightblue', alpha=0.6, label='Episode reward')
plt.plot(ma_x, ma, color='blue', linewidth=2, label=f'MA({window})')
plt.title('SAC Training Convergence (AAPL seed42)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

out_path = os.path.join(vis_dir, 'sac_convergence_AAPL_seed42_regen.png')
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print('Saved plot to', out_path)

# Try to open on Windows
try:
    if os.name == 'nt':
        os.startfile(out_path)
except Exception as e:
    print('Unable to open file automatically:', e)
