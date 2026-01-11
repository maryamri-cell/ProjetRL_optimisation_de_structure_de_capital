"""Train SAC and generate convergence curves for ALL companies
"""
import os
import sys
import argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config


def load_all_real_data():
    """Load all real company data"""
    dataset_path = os.path.join('data', 'training', 'real_data_dataset.npy')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)
    dataset = np.load(dataset_path, allow_pickle=True).item()
    return dataset


def make_env(config, company_data, seed, log_path=None, no_reward_normalization=False):
    def _init():
        env = CapitalStructureEnv(
            config=config,
            max_steps=min(252, len(company_data)),
            scenario='baseline',
            real_cf_data=company_data,
            disable_reward_normalization=no_reward_normalization
        )
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            env = Monitor(env, filename=log_path)
        try:
            env.reset(seed=seed)
        except:
            env.reset()
        return env
    return _init


def train_sac_single(ticker, company_data, config, timesteps, seed, no_reward_normalization=True):
    """Train SAC on a single company"""
    
    print(f"\n{'='*70}")
    print(f"Training SAC on {ticker} ({timesteps:,} timesteps)")
    print(f"{'='*70}")
    
    company_cf = company_data['cf_normalized']
    
    # Load hyperparams
    with open('configs/optimized_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
    algo_params = hyperparams['SAC']
    
    # Setup logging
    log_dir = os.path.join('logs', 'convergence', ticker, f'SAC_all_companies_seed{seed}')
    os.makedirs(log_dir, exist_ok=True)
    monitor_file = os.path.join(log_dir, 'monitor.csv')
    
    # Create environment
    env_fn = make_env(config, company_cf, seed, monitor_file, no_reward_normalization)
    env = SubprocVecEnv([env_fn])
    
    # Create SAC model
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=algo_params.get('learning_rate', 1e-4),
        buffer_size=algo_params.get('buffer_size', 100_000),
        learning_starts=algo_params.get('learning_starts', 1_000),
        batch_size=algo_params.get('batch_size', 64),
        tau=algo_params.get('tau', 0.02),
        gamma=algo_params.get('gamma', 0.99),
        train_freq=algo_params.get('train_freq', 4),
        gradient_steps=algo_params.get('gradient_steps', 4),
        ent_coef=algo_params.get('ent_coef', 0.2),
        target_entropy=algo_params.get('target_entropy', -3),
        verbose=0,  # No verbose output
        tensorboard_log=os.path.join(log_dir, 'tensorboard')
    )
    
    # Train
    print(f"Training in progress...")
    model.learn(total_timesteps=timesteps, progress_bar=False)
    
    # Save model
    model_path = os.path.join('models', 'real_data', ticker, f'SAC_all_companies_seed{seed}')
    os.makedirs(model_path, exist_ok=True)
    model.save(os.path.join(model_path, 'final_model'))
    
    env.close()
    
    # Parse monitor.csv
    rewards = []
    with open(monitor_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                r = float(row.get('r', row.get('reward', 0)))
                rewards.append(r)
            except:
                pass
    
    # Save episode rewards CSV
    out_csv = os.path.join(log_dir, 'episode_rewards.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward'])
        for i, r in enumerate(rewards, 1):
            writer.writerow([i, r])
    
    mean_reward = np.mean(rewards) if rewards else 0
    std_reward = np.std(rewards) if rewards else 0
    
    print(f"✓ Training complete: {ticker}")
    print(f"  Avg Reward: {mean_reward:.4f} | Std: {std_reward:.4f} | Episodes: {len(rewards)}")
    
    return rewards, log_dir, mean_reward, std_reward


def plot_individual_convergence(ticker, rewards, log_dir):
    """Plot individual company convergence"""
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.5, linewidth=0.8, label='Episode reward')
    
    # Add moving average
    window = 20
    if len(rewards) > window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(ma, linewidth=2, label=f'MA({window})', color='green')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'SAC Convergence - {ticker}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    out_png = os.path.join('visualizations', f'sac_convergence_{ticker}_all_companies.png')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    return out_png


def plot_all_companies_comparison(results_dict):
    """Create comprehensive comparison plot across all companies"""
    
    num_companies = len(results_dict)
    
    # Determine grid size
    cols = min(4, num_companies)
    rows = (num_companies + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    if num_companies == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    stats = []
    
    for idx, (ticker, (rewards, mean, std)) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        # Plot individual rewards
        ax.plot(rewards, alpha=0.4, linewidth=0.8)
        
        # Add moving average
        window = 20
        if len(rewards) > window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(ma, linewidth=2, color='green')
        
        ax.set_title(f'{ticker}\nμ={mean:.4f}, σ={std:.4f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        
        stats.append({
            'ticker': ticker,
            'mean': mean,
            'std': std,
            'episodes': len(rewards)
        })
    
    # Remove empty subplots
    for idx in range(num_companies, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    out_png = os.path.join('visualizations', 'sac_convergence_all_companies.png')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    return out_png, stats


def plot_summary_statistics(stats):
    """Plot summary statistics across all companies"""
    
    tickers = [s['ticker'] for s in stats]
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot of mean rewards
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(tickers)))
    axes[0].bar(tickers, means, color=colors, edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Mean Reward by Company')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=np.mean(means), color='red', linestyle='--', linewidth=2, label='Overall Mean')
    axes[0].legend()
    
    # Error bar plot
    axes[1].errorbar(range(len(tickers)), means, yerr=stds, fmt='o', markersize=8, capsize=5)
    axes[1].set_xticks(range(len(tickers)))
    axes[1].set_xticklabels(tickers, rotation=45)
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Mean ± Std by Company')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_png = os.path.join('visualizations', 'sac_summary_statistics.png')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    return out_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps per company')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--tickers', type=str, nargs='*', default=None, help='Specific tickers to train (default: all)')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Load all data
    print("📊 Loading all real company data...")
    dataset = load_all_real_data()
    all_tickers = list(dataset.keys())
    
    # Filter tickers if specified
    if args.tickers:
        tickers_to_train = [t for t in args.tickers if t in all_tickers]
    else:
        tickers_to_train = all_tickers[:10]  # Default: first 10 companies
    
    print(f"✓ Will train {len(tickers_to_train)} companies: {', '.join(tickers_to_train)}")
    
    config = load_config('config.yaml')
    
    # Train SAC on each company
    results_dict = {}
    
    for i, ticker in enumerate(tickers_to_train, 1):
        try:
            company_data = dataset[ticker]
            print(f"\n[{i}/{len(tickers_to_train)}] Processing {ticker}...")
            
            rewards, log_dir, mean_reward, std_reward = train_sac_single(
                ticker=ticker,
                company_data=company_data,
                config=config,
                timesteps=args.timesteps,
                seed=args.seed,
                no_reward_normalization=True
            )
            
            # Generate individual plot
            plot_png = plot_individual_convergence(ticker, rewards, log_dir)
            print(f"  Plot: {plot_png}")
            
            results_dict[ticker] = (rewards, mean_reward, std_reward)
            
        except Exception as e:
            print(f"  ⚠ Error training {ticker}: {e}")
            continue
    
    # Generate comparison plots
    print(f"\n{'='*70}")
    print("Generating comparison plots...")
    print(f"{'='*70}")
    
    if results_dict:
        # All companies grid
        comp_png, stats = plot_all_companies_comparison(results_dict)
        print(f"✓ Comparison plot: {comp_png}")
        
        # Summary statistics
        summary_png = plot_summary_statistics(stats)
        print(f"✓ Summary statistics: {summary_png}")
        
        # Print summary table
        print(f"\n{'TICKER':<10} {'MEAN':<10} {'STD':<10} {'EPISODES':<10}")
        print(f"{'-'*40}")
        for s in stats:
            print(f"{s['ticker']:<10} {s['mean']:<10.4f} {s['std']:<10.4f} {s['episodes']:<10}")
        
        overall_mean = np.mean([s['mean'] for s in stats])
        overall_std = np.mean([s['std'] for s in stats])
        print(f"{'-'*40}")
        print(f"{'OVERALL':<10} {overall_mean:<10.4f} {overall_std:<10.4f}")
    
    print(f"\n{'='*70}")
    print("✅ All companies trained and plots generated!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
