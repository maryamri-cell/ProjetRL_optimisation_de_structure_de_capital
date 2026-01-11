"""Train SAC with improved hyperparams and plot convergence comparison
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config
from src.utils.prioritized_replay_buffer import PrioritizedReplayBuffer


def load_real_data(ticker):
    dataset_path = os.path.join('data', 'training', 'real_data_dataset.npy')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)
    dataset = np.load(dataset_path, allow_pickle=True).item()
    return dataset[ticker]


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


def train_sac(ticker, timesteps, seed, use_prioritized=True):
    """Train SAC with improved hyperparams"""
    
    print(f"\n{'='*80}")
    print(f"Training SAC on {ticker} ({timesteps:,} timesteps)")
    print(f"Use Prioritized Replay: {use_prioritized}")
    print(f"{'='*80}")
    
    config = load_config('config.yaml')
    company = load_real_data(ticker)
    company_cf = company['cf_normalized']
    
    # Load hyperparams
    with open('configs/optimized_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
    algo_params = hyperparams['SAC']
    
    # Setup logging
    log_dir = os.path.join('logs', 'convergence', ticker, f'SAC_improved_seed{seed}')
    os.makedirs(log_dir, exist_ok=True)
    monitor_file = os.path.join(log_dir, 'monitor.csv')
    
    # Create environment
    env_fn = make_env(config, company_cf, seed, monitor_file, no_reward_normalization=True)
    env = SubprocVecEnv([env_fn])
    
    # Create SAC model with new hyperparams
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
        verbose=1,
        tensorboard_log=os.path.join(log_dir, 'tensorboard')
    )
    
    # Use prioritized replay buffer if requested (disabled for now - compatibility issues)
    # TODO: Fix PrioritizedReplayBuffer to work with stable-baselines3 SAC
    if use_prioritized and False:  # Disabled
        try:
            model.replay_buffer = PrioritizedReplayBuffer(
                buffer_size=algo_params.get('buffer_size', 100_000),
                observation_space=model.observation_space,
                action_space=model.action_space,
                device=model.device,
                n_envs=1,
                alpha=0.6,
                beta=0.4
            )
            print("✓ Prioritized Replay Buffer enabled")
        except Exception as e:
            print(f"⚠ Prioritized buffer skipped (will use standard buffer)")
            print(f"  Reason: {type(e).__name__}")
    else:
        if use_prioritized:
            print("ℹ Prioritized replay buffer disabled (compatibility under development)")
    
    # Train
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    # Save model
    model_path = os.path.join('models', 'real_data', ticker, f'SAC_improved_seed{seed}')
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
    
    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Monitor saved to: {monitor_file}")
    print(f"✓ Episode CSV saved to: {out_csv}")
    print(f"  Avg reward: {np.mean(rewards):.4f}")
    print(f"  Std reward: {np.std(rewards):.4f}")
    
    return rewards, log_dir


def plot_comparison(baseline_rewards, improved_rewards, ticker, seed):
    """Plot baseline vs improved convergence"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Both overlaid with moving average
    axes[0].plot(baseline_rewards, alpha=0.3, label='Baseline (seed=42)', color='blue')
    axes[0].plot(improved_rewards, alpha=0.3, label='Improved (seed=42)', color='green')
    
    # Moving average
    window = 20
    if len(baseline_rewards) > window:
        baseline_ma = np.convolve(baseline_rewards, np.ones(window)/window, mode='valid')
        improved_ma = np.convolve(improved_rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(baseline_ma, linewidth=2, label=f'Baseline MA({window})', color='blue')
        axes[0].plot(improved_ma, linewidth=2, label=f'Improved MA({window})', color='green')
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title(f'SAC Convergence Comparison ({ticker})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Improvement over baseline
    improvement = np.array(improved_rewards) - np.array(baseline_rewards)
    axes[1].plot(improvement, alpha=0.5, color='purple')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    if len(improvement) > window:
        improvement_ma = np.convolve(improvement, np.ones(window)/window, mode='valid')
        axes[1].plot(improvement_ma, linewidth=2, label=f'Improvement MA({window})', color='purple')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Reward Difference (Improved - Baseline)')
    axes[1].set_title(f'Improvement over Baseline')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Save
    out_png = os.path.join('visualizations', f'sac_convergence_comparison_{ticker}_seed{seed}.png')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\n✓ Comparison plot saved to: {out_png}")
    
    return out_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL', help='Ticker symbol')
    parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Train improved SAC
    improved_rewards, log_dir = train_sac(
        ticker=args.ticker,
        timesteps=args.timesteps,
        seed=args.seed,
        use_prioritized=True
    )
    
    # Load baseline from previous run if available
    baseline_file = os.path.join('logs', 'convergence', args.ticker, 'SAC_seed42', 'episode_rewards.csv')
    baseline_rewards = None
    
    if os.path.exists(baseline_file):
        print(f"\n✓ Found baseline file: {baseline_file}")
        with open(baseline_file, 'r') as f:
            reader = csv.DictReader(f)
            baseline_rewards = []
            for row in reader:
                try:
                    r = float(row.get('reward', 0))
                    baseline_rewards.append(r)
                except:
                    pass
        
        if baseline_rewards and len(baseline_rewards) > 0:
            # Pad to same length for comparison
            max_len = max(len(baseline_rewards), len(improved_rewards))
            baseline_padded = list(baseline_rewards) + [baseline_rewards[-1]] * (max_len - len(baseline_rewards))
            improved_padded = list(improved_rewards) + [improved_rewards[-1]] * (max_len - len(improved_rewards))
            
            plot_comparison(baseline_padded[:max_len], improved_padded[:max_len], args.ticker, args.seed)
            
            print(f"\nBaseline Stats:")
            print(f"  Mean: {np.mean(baseline_rewards):.4f}")
            print(f"  Std:  {np.std(baseline_rewards):.4f}")
            print(f"\nImproved Stats:")
            print(f"  Mean: {np.mean(improved_rewards):.4f}")
            print(f"  Std:  {np.std(improved_rewards):.4f}")
            print(f"\nDifference:")
            print(f"  Mean improvement: {np.mean(improved_rewards) - np.mean(baseline_rewards):+.4f}")
    else:
        print(f"\n⚠ Baseline file not found at: {baseline_file}")
        print("   Plot comparison skipped")


if __name__ == '__main__':
    main()
