"""Train TD3 on real data and generate convergence plot"""
import os
import sys
import argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config


def load_real_data(ticker):
    """Load real data for a specific ticker"""
    dataset_path = os.path.join('data', 'training', 'real_data_dataset.npy')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)
    dataset = np.load(dataset_path, allow_pickle=True).item()
    return dataset[ticker]


def make_env(config, company_data, seed, log_path=None):
    """Factory function to create environment"""
    def _init():
        env = CapitalStructureEnv(
            config=config,
            max_steps=min(252, len(company_data)),
            scenario='baseline',
            real_cf_data=company_data,
            disable_reward_normalization=False
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


def train_td3(ticker='AAPL', timesteps=100_000, seed=42):
    """Train TD3 on real data"""
    
    print(f"\n{'='*80}")
    print(f"Training TD3 on {ticker} ({timesteps:,} timesteps)")
    print(f"Seed: {seed}")
    print(f"{'='*80}")
    
    config = load_config('config.yaml')
    company = load_real_data(ticker)
    company_cf = company['cf_normalized']
    
    print(f"Company CF shape: {company_cf.shape}")
    print(f"Company CF mean: {np.mean(company_cf):.4f}")
    
    # Setup logging
    log_dir = os.path.join('logs', 'convergence', ticker, f'TD3_seed{seed}')
    os.makedirs(log_dir, exist_ok=True)
    monitor_file = os.path.join(log_dir, 'monitor.csv')
    
    # Create environment
    env_fn = make_env(config, company_cf, seed, monitor_file)
    env = SubprocVecEnv([env_fn])
    
    # Create TD3 model with standard hyperparams
    # TD3 = Twin Delayed DDPG with better exploration/stability
    model = TD3(
        'MlpPolicy',
        env,
        learning_rate=1e-3,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_delay=2,
        action_noise=None,  # Will use OU noise by default
        verbose=1,
        tensorboard_log=os.path.join(log_dir, 'tensorboard')
    )
    
    print("\nStarting training...")
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    # Save model
    model_path = os.path.join('models', 'real_data', ticker, f'TD3_seed{seed}')
    os.makedirs(model_path, exist_ok=True)
    model.save(os.path.join(model_path, 'final_model'))
    print(f"✓ Model saved to: {model_path}")
    
    env.close()
    
    # Parse monitor.csv and extract rewards (skip JSON header)
    rewards = []
    with open(monitor_file, 'r') as f:
        lines = f.readlines()
        # Skip the first line if it contains JSON metadata
        start_idx = 1 if lines and lines[0].startswith('#') else 0
        reader = csv.DictReader(lines[start_idx:])
        for row in reader:
            try:
                r = float(row.get('r', 0))
                rewards.append(r)
            except (ValueError, TypeError, AttributeError):
                pass
    
    # Save episode rewards CSV
    out_csv = os.path.join(log_dir, 'episode_rewards.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward'])
        for i, r in enumerate(rewards, 1):
            writer.writerow([i, r])
    
    print(f"\n{'='*80}")
    print(f"✓ Training complete!")
    print(f"✓ Monitor saved to: {monitor_file}")
    print(f"✓ Episode CSV saved to: {out_csv}")
    print(f"  Episodes completed: {len(rewards)}")
    if rewards:
        print(f"  Avg reward: {np.mean(rewards):.4f}")
        print(f"  Std reward: {np.std(rewards):.4f}")
        print(f"  Min reward: {np.min(rewards):.4f}")
        print(f"  Max reward: {np.max(rewards):.4f}")
    print(f"{'='*80}\n")
    
    return rewards, log_dir


def plot_convergence(rewards, ticker, seed, log_dir):
    """Generate convergence plot"""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Raw rewards
    axes[0].plot(rewards, alpha=0.5, color='blue', linewidth=0.5)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title(f'TD3 Convergence - Raw Rewards ({ticker}, seed={seed})')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Moving average
    window = 20
    if len(rewards) > window:
        rewards_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[1].plot(rewards_ma, color='green', linewidth=2, label=f'MA({window})')
        axes[1].fill_between(range(len(rewards_ma)), rewards_ma, alpha=0.3, color='green')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Reward (Moving Avg)')
        axes[1].set_title(f'TD3 Convergence - Moving Average ({ticker}, seed={seed})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join('visualizations', f'td3_convergence_{ticker.lower()}_seed{seed}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Convergence plot saved to: {plot_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train TD3 on real data')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Company ticker')
    parser.add_argument('--timesteps', type=int, default=100_000, help='Total timesteps to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    # Train
    rewards, log_dir = train_td3(
        ticker=args.ticker,
        timesteps=args.timesteps,
        seed=args.seed
    )
    
    # Plot
    if not args.no_plot:
        plot_convergence(rewards, args.ticker, args.seed, log_dir)


if __name__ == '__main__':
    main()
