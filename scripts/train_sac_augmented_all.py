"""Train SAC on augmented dataset and generate convergence curves for all companies
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

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config


def load_augmented_data():
    """Load augmented real company data"""
    dataset_path = os.path.join('data', 'training', 'real_data_dataset_augmented_20.npy')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Augmented dataset not found: {dataset_path}")
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


def train_sac_augmented(ticker, company_data, config, timesteps, seed, no_reward_normalization=True):
    """Train SAC on augmented company data"""
    
    print(f"\n{'─'*70}")
    print(f"Training SAC on {ticker} (Augmented, {timesteps:,} timesteps across all variants)")
    print(f"{'─'*70}")
    
    # Get normalized CF data - shape is (n_quarters, n_variants) for augmented data
    cf_normalized = company_data['cf_normalized']
    
    # Check if it's augmented (2D) or single (1D)
    if cf_normalized.ndim == 2:
        n_quarters, n_variants = cf_normalized.shape
        print(f"  Augmented dataset: {n_variants} variants × {n_quarters} quarters")
        # Concatenate all variants into a single sequence for training
        # Shape will be (n_quarters * n_variants,)
        company_cf = cf_normalized.flatten()
    else:
        company_cf = cf_normalized
    
    # Load hyperparams
    with open('configs/optimized_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
    algo_params = hyperparams['SAC']
    
    # Setup logging
    log_dir = os.path.join('logs', 'convergence', ticker, f'SAC_augmented_seed{seed}')
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
        verbose=0,
        tensorboard_log=os.path.join(log_dir, 'tensorboard')
    )
    
    # Train
    print(f"Training in progress...")
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    # Save model
    model_path = os.path.join('models', 'real_data', ticker, f'SAC_augmented_seed{seed}')
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
    
    print(f"✅ {ticker}: μ={mean_reward:.4f} | σ={std_reward:.4f} | episodes={len(rewards)}")
    
    return rewards, log_dir, mean_reward, std_reward


def plot_individual_convergence(ticker, rewards, log_dir):
    """Plot individual company convergence"""
    
    fig, ax = plt.subplots(figsize=(11, 5))
    
    # Raw rewards
    ax.plot(rewards, alpha=0.4, linewidth=0.8, color='steelblue', label='Episode reward')
    
    # Moving average
    window = 20
    if len(rewards) > window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(ma, linewidth=2.5, color='darkgreen', label=f'MA({window})')
    
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title(f'SAC Convergence - {ticker} (Augmented Dataset)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    out_png = os.path.join('visualizations', f'sac_convergence_{ticker}_augmented.png')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    return out_png


def plot_all_companies_grid(results_dict):
    """Create grid plot across all companies"""
    
    num_companies = len(results_dict)
    
    # Determine grid size (aim for ~4 columns)
    cols = 4
    rows = (num_companies + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5*rows))
    if num_companies == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    stats = []
    
    for idx, (ticker, (rewards, mean, std)) in enumerate(sorted(results_dict.items())):
        ax = axes[idx]
        
        # Raw rewards
        ax.plot(rewards, alpha=0.3, linewidth=0.7, color='lightblue')
        
        # Moving average
        window = 20
        if len(rewards) > window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(ma, linewidth=2.5, color='green')
        
        # Title with stats
        title = f'{ticker}\nμ={mean:.4f} | σ={std:.4f}'
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel('Reward', fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.tick_params(labelsize=8)
        
        stats.append({'ticker': ticker, 'mean': mean, 'std': std, 'episodes': len(rewards)})
    
    # Remove empty subplots
    for idx in range(num_companies, len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle('SAC Convergence across All Companies (Augmented Dataset)', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    out_png = os.path.join('visualizations', 'sac_convergence_grid_augmented.png')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    return out_png, stats


def plot_summary_bar_chart(stats):
    """Plot bar chart of rewards by company"""
    
    tickers = [s['ticker'] for s in stats]
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of means
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(tickers)))
    bars = axes[0].bar(tickers, means, color=colors, edgecolor='black', alpha=0.8, linewidth=1.2)
    axes[0].set_ylabel('Average Reward', fontsize=11, fontweight='bold')
    axes[0].set_title('Mean Reward by Company', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[0].axhline(y=np.mean(means), color='red', linestyle='--', linewidth=2.5, 
                    label=f'Overall Mean: {np.mean(means):.4f}')
    axes[0].legend(fontsize=10)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    # Error bar plot
    axes[1].errorbar(range(len(tickers)), means, yerr=stds, fmt='o', 
                    markersize=8, capsize=5, linewidth=1.5, color='steelblue')
    axes[1].set_xticks(range(len(tickers)))
    axes[1].set_xticklabels(tickers, rotation=45)
    axes[1].set_ylabel('Reward', fontsize=11, fontweight='bold')
    axes[1].set_title('Mean ± Std by Company', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    out_png = os.path.join('visualizations', 'sac_summary_augmented.png')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    return out_png


def main():
    parser = argparse.ArgumentParser(description='Train SAC on augmented dataset for all companies')
    parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps per company')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--tickers', type=str, nargs='*', default=None, help='Specific tickers (default: all)')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Load augmented dataset
    print("\n" + "="*70)
    print("📊 Loading Augmented Real Dataset")
    print("="*70)
    dataset = load_augmented_data()
    all_tickers = sorted(list(dataset.keys()))
    
    print(f"✅ Loaded {len(all_tickers)} companies from augmented dataset")
    print(f"   Companies: {', '.join(all_tickers[:10])}{'...' if len(all_tickers) > 10 else ''}")
    
    # Filter tickers if specified
    if args.tickers:
        tickers_to_train = [t for t in args.tickers if t in all_tickers]
    else:
        tickers_to_train = all_tickers
    
    print(f"\n📈 Will train {len(tickers_to_train)} companies")
    print(f"   Timesteps per company: {args.timesteps:,}")
    print(f"   Random seed: {args.seed}")
    
    config = load_config('config.yaml')
    
    # Train SAC on each company
    results_dict = {}
    successful_tickers = []
    failed_tickers = []
    
    print("\n" + "="*70)
    print("🚀 Starting Training")
    print("="*70)
    
    for i, ticker in enumerate(tickers_to_train, 1):
        try:
            company_data = dataset[ticker]
            print(f"\n[{i:2d}/{len(tickers_to_train)}] {ticker:<6}", end=" ")
            
            rewards, log_dir, mean_reward, std_reward = train_sac_augmented(
                ticker=ticker,
                company_data=company_data,
                config=config,
                timesteps=args.timesteps,
                seed=args.seed,
                no_reward_normalization=True
            )
            
            # Generate individual plot
            plot_png = plot_individual_convergence(ticker, rewards, log_dir)
            
            results_dict[ticker] = (rewards, mean_reward, std_reward)
            successful_tickers.append(ticker)
            
        except Exception as e:
            print(f"[{i:2d}/{len(tickers_to_train)}] {ticker:<6} ❌ Error: {str(e)[:50]}")
            failed_tickers.append(ticker)
            continue
    
    # Generate comparison plots
    print(f"\n{'='*70}")
    print("📊 Generating Comparison Plots")
    print(f"{'='*70}")
    
    if results_dict:
        # Grid plot
        grid_png, stats = plot_all_companies_grid(results_dict)
        print(f"✅ Grid plot saved: visualizations/sac_convergence_grid_augmented.png")
        
        # Summary bar chart
        summary_png = plot_summary_bar_chart(stats)
        print(f"✅ Summary chart saved: visualizations/sac_summary_augmented.png")
        
        # Print summary table
        print(f"\n{'='*70}")
        print(f"{'TICKER':<10} {'MEAN':<12} {'STD':<12} {'EPISODES':<12}")
        print(f"{'─'*70}")
        
        for s in sorted(stats, key=lambda x: x['mean'], reverse=True):
            print(f"{s['ticker']:<10} {s['mean']:<12.4f} {s['std']:<12.4f} {s['episodes']:<12}")
        
        # Overall stats
        overall_mean = np.mean([s['mean'] for s in stats])
        overall_std = np.mean([s['std'] for s in stats])
        overall_max = max([s['mean'] for s in stats])
        overall_min = min([s['mean'] for s in stats])
        
        print(f"{'─'*70}")
        print(f"{'MEAN':<10} {overall_mean:<12.4f}")
        print(f"{'STD':<10} {overall_std:<12.4f}")
        print(f"{'MAX':<10} {overall_max:<12.4f}")
        print(f"{'MIN':<10} {overall_min:<12.4f}")
        
        print(f"\n{'='*70}")
        print(f"✅ Trained {len(successful_tickers)}/{len(tickers_to_train)} companies successfully")
        if failed_tickers:
            print(f"⚠️  Failed: {', '.join(failed_tickers)}")
        print(f"{'='*70}\n")
    else:
        print(f"⚠️  No companies trained successfully")


if __name__ == '__main__':
    main()
