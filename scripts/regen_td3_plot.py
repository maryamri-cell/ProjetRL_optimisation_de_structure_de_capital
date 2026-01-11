"""Regenerate TD3-only convergence plot from episode_rewards.csv"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = 'logs/convergence/AAPL/TD3_seed42/episode_rewards.csv'
OUT = 'visualizations/td3_convergence_aapl_seed42_regen.png'

rewards = []
with open(LOG, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            rewards.append(float(row.get('reward', row.get('r', 0))))
        except:
            pass

if not rewards:
    print('No rewards found in', LOG)
else:
    plt.figure(figsize=(10,6))
    plt.plot(rewards, alpha=0.5, linewidth=0.6, label='TD3 rewards')
    window = 20
    if len(rewards) > window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, window-1+len(ma)), ma, color='red', linewidth=2, label=f'MA({window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('TD3 Convergence (AAPL, seed=42)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    plt.close()
    print('Saved:', OUT)
