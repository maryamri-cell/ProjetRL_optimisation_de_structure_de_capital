"""Regenerate convergence plots for AUGMENTED runs (PPO, SAC, TD3)
Reads episode_rewards.csv for each algorithm and seed, saves per-algo and comparison plots.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_rewards(path):
    rewards = []
    try:
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    r = float(row.get('reward', row.get('r', 0)))
                    rewards.append(r)
                except:
                    pass
    except FileNotFoundError:
        print(f"Warning: {path} not found")
    return rewards


def plot_single(rewards, out_path, title=None, window=20):
    if not rewards:
        print('No rewards to plot for', out_path)
        return
    plt.figure(figsize=(10,6))
    plt.plot(rewards, alpha=0.4, linewidth=0.6, label='raw')
    if len(rewards) > window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, window-1+len(ma)), ma, color='red', linewidth=2, label=f'MA({window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved:', out_path)


def plot_comparison(reward_dict, out_path, ticker, seed, window=20):
    plt.figure(figsize=(12,6))
    for algo, rewards in reward_dict.items():
        if not rewards:
            continue
        if len(rewards) > window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, window-1+len(ma)), ma, linewidth=2, label=f'{algo} MA({window})')
        else:
            plt.plot(rewards, alpha=0.4, linewidth=0.6, label=f'{algo} raw')
    plt.xlabel('Episode')
    plt.ylabel('Reward (moving avg)')
    plt.title(f'Algorithm Comparison - {ticker} (seed={seed})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved:', out_path)


if __name__ == '__main__':
    ticker = 'AUGMENTED'
    seed = 1
    algos = ['PPO', 'SAC', 'TD3']
    reward_dict = {}

    for algo in algos:
        csv_path = os.path.join('logs', 'convergence', ticker, f'{algo}_aug_no_norm_seed{seed}', 'episode_rewards.csv')
        rewards = load_rewards(csv_path)
        reward_dict[algo] = rewards
        out = os.path.join('visualizations', f'{algo.lower()}_aug_no_norm_seed{seed}.png')
        title = f'{algo} Convergence - {ticker} (seed={seed})'
        plot_single(rewards, out, title=title)

    # Combined comparison
    out_cmp = os.path.join('visualizations', f'comparison_aug_seed{seed}.png')
    plot_comparison(reward_dict, out_cmp, ticker, seed)
