"""Compare convergence curves for SAC, PPO, and TD3 algorithms"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_episode_rewards(log_path):
    """Load episode rewards from CSV file"""
    rewards = []
    try:
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    r = float(row.get('reward', 0))
                    rewards.append(r)
                except:
                    pass
    except FileNotFoundError:
        print(f"Warning: {log_path} not found")
    return rewards


def plot_algorithm_comparison(ticker='AAPL', seed=42):
    """Create comparison plot of all three algorithms"""
    
    # Load rewards for each algorithm
    sac_rewards = load_episode_rewards(f'logs/convergence/{ticker}/SAC_seed{seed}/episode_rewards.csv')
    td3_rewards = load_episode_rewards(f'logs/convergence/{ticker}/TD3_seed{seed}/episode_rewards.csv')
    ppo_rewards = load_episode_rewards(f'logs/convergence/{ticker}/PPO_seed{seed}/episode_rewards.csv')
    
    print(f"Loaded rewards:")
    print(f"  SAC: {len(sac_rewards)} episodes")
    print(f"  TD3: {len(td3_rewards)} episodes")
    print(f"  PPO: {len(ppo_rewards)} episodes")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All three algorithms overlaid (raw)
    ax = axes[0, 0]
    if sac_rewards:
        ax.plot(sac_rewards, alpha=0.3, color='blue', linewidth=0.5, label=f'SAC ({len(sac_rewards)} ep)')
    if td3_rewards:
        ax.plot(td3_rewards, alpha=0.3, color='red', linewidth=0.5, label=f'TD3 ({len(td3_rewards)} ep)')
    if ppo_rewards:
        ax.plot(ppo_rewards, alpha=0.3, color='green', linewidth=0.5, label=f'PPO ({len(ppo_rewards)} ep)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'Algorithm Comparison - Raw Rewards ({ticker}, seed={seed})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Moving average comparison
    ax = axes[0, 1]
    window = 20
    if len(sac_rewards) > window:
        sac_ma = np.convolve(sac_rewards, np.ones(window)/window, mode='valid')
        ax.plot(sac_ma, color='blue', linewidth=2, label=f'SAC MA({window})')
    if len(td3_rewards) > window:
        td3_ma = np.convolve(td3_rewards, np.ones(window)/window, mode='valid')
        ax.plot(td3_ma, color='red', linewidth=2, label=f'TD3 MA({window})')
    if len(ppo_rewards) > window:
        ppo_ma = np.convolve(ppo_rewards, np.ones(window)/window, mode='valid')
        ax.plot(ppo_ma, color='green', linewidth=2, label=f'PPO MA({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (Moving Avg)')
    ax.set_title(f'Algorithm Comparison - Moving Average ({window} window)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Statistics comparison
    ax = axes[1, 0]
    algorithms = []
    means = []
    stds = []
    colors = []
    
    if sac_rewards:
        algorithms.append('SAC')
        means.append(np.mean(sac_rewards))
        stds.append(np.std(sac_rewards))
        colors.append('blue')
    if td3_rewards:
        algorithms.append('TD3')
        means.append(np.mean(td3_rewards))
        stds.append(np.std(td3_rewards))
        colors.append('red')
    if ppo_rewards:
        algorithms.append('PPO')
        means.append(np.mean(ppo_rewards))
        stds.append(np.std(ppo_rewards))
        colors.append('green')
    
    x_pos = np.arange(len(algorithms))
    ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, error_kw={'linewidth': 2})
    ax.set_ylabel('Mean Reward')
    ax.set_title(f'Mean Rewards with Std Dev ({ticker})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [['Algorithm', 'Episodes', 'Mean', 'Std', 'Min', 'Max']]
    
    if sac_rewards:
        table_data.append([
            'SAC',
            f'{len(sac_rewards)}',
            f'{np.mean(sac_rewards):.4f}',
            f'{np.std(sac_rewards):.4f}',
            f'{np.min(sac_rewards):.4f}',
            f'{np.max(sac_rewards):.4f}'
        ])
    
    if td3_rewards:
        table_data.append([
            'TD3',
            f'{len(td3_rewards)}',
            f'{np.mean(td3_rewards):.4f}',
            f'{np.std(td3_rewards):.4f}',
            f'{np.min(td3_rewards):.4f}',
            f'{np.max(td3_rewards):.4f}'
        ])
    
    if ppo_rewards:
        table_data.append([
            'PPO',
            f'{len(ppo_rewards)}',
            f'{np.mean(ppo_rewards):.4f}',
            f'{np.std(ppo_rewards):.4f}',
            f'{np.min(ppo_rewards):.4f}',
            f'{np.max(ppo_rewards):.4f}'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title(f'Performance Statistics ({ticker})', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join('visualizations', f'comparison_sac_td3_ppo_{ticker.lower()}_seed{seed}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {plot_path}")
    
    plt.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ALGORITHM COMPARISON SUMMARY ({ticker}, seed={seed})")
    print(f"{'='*80}")
    
    if sac_rewards:
        print(f"\nSAC:")
        print(f"  Episodes: {len(sac_rewards)}")
        print(f"  Mean:     {np.mean(sac_rewards):.6f}")
        print(f"  Std:      {np.std(sac_rewards):.6f}")
        print(f"  Min:      {np.min(sac_rewards):.6f}")
        print(f"  Max:      {np.max(sac_rewards):.6f}")
    
    if td3_rewards:
        print(f"\nTD3:")
        print(f"  Episodes: {len(td3_rewards)}")
        print(f"  Mean:     {np.mean(td3_rewards):.6f}")
        print(f"  Std:      {np.std(td3_rewards):.6f}")
        print(f"  Min:      {np.min(td3_rewards):.6f}")
        print(f"  Max:      {np.max(td3_rewards):.6f}")
    
    if ppo_rewards:
        print(f"\nPPO:")
        print(f"  Episodes: {len(ppo_rewards)}")
        print(f"  Mean:     {np.mean(ppo_rewards):.6f}")
        print(f"  Std:      {np.std(ppo_rewards):.6f}")
        print(f"  Min:      {np.min(ppo_rewards):.6f}")
        print(f"  Max:      {np.max(ppo_rewards):.6f}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    plot_algorithm_comparison(ticker='AAPL', seed=42)
