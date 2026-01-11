"""Generate SAC convergence curve (quick smoke training + plot)

Creates a Monitor-wrapped single environment, trains SAC for a small
number of timesteps, saves the Monitor CSV, and plots episode rewards.
"""
import os
import argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config


def load_real_data(ticker=None):
    dataset_path = os.path.join('data', 'training', 'real_data_dataset.npy')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)
    dataset = np.load(dataset_path, allow_pickle=True).item()
    if ticker:
        return dataset[ticker]
    return dataset


def make_monitored_env(config, company_data, seed, logfile_path, no_reward_normalization=False):
    def _init():
        env = CapitalStructureEnv(
            config=config,
            max_steps=min(252, len(company_data)),
            scenario='baseline',
            real_cf_data=company_data,
            disable_reward_normalization=no_reward_normalization
        )
        os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
        env = Monitor(env, filename=logfile_path)
        try:
            env.reset(seed=seed)
        except Exception:
            env.reset()
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-reward-normalization', action='store_true')
    args = parser.parse_args()

    config = load_config('config.yaml')
    company = load_real_data(args.ticker)
    company_cf = company['cf_normalized']

    log_dir = os.path.join('logs', 'convergence', args.ticker, f'SAC_seed{args.seed}')
    os.makedirs(log_dir, exist_ok=True)
    monitor_file = os.path.join(log_dir, 'monitor.csv')

    env_fn = make_monitored_env(config, company_cf, args.seed, monitor_file, args.no_reward_normalization)
    env = DummyVecEnv([env_fn])

    # Simple hyperparams — keep defaults consistent with project
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=os.path.join(log_dir, 'tensorboard'))

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    model_path = os.path.join('models', 'real_data', args.ticker, f'SAC_seed{args.seed}', 'final_model_quick')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    env.close()

    # Parse monitor.csv
    if not os.path.exists(monitor_file):
        print('Monitor file not found:', monitor_file)
        return

    import csv
    rewards = []
    lengths = []
    times = []
    with open(monitor_file, 'r') as f:
        # Skip comments
        header = None
        for line in f:
            if line.startswith('#'):
                continue
            if header is None:
                header = line.strip().split(',')
                continue
            vals = line.strip().split(',')
            row = dict(zip(header, vals))
            rewards.append(float(row.get('r', row.get('reward', 0))))
            lengths.append(int(float(row.get('l', row.get('length', 0)))))
            times.append(float(row.get('t', row.get('time', 0))))

    # Save CSV of episode rewards for reference
    out_csv = os.path.join(log_dir, 'episode_rewards.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'length', 'time'])
        for i, (r, l, t) in enumerate(zip(rewards, lengths, times), start=1):
            writer.writerow([i, r, l, t])

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(rewards, label='Episode reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'SAC convergence ({args.ticker}, seed={args.seed})')
    plt.grid(True)
    plt.legend()

    out_png = os.path.join('visualizations', f'sac_convergence_{args.ticker}_seed{args.seed}.png')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)

    print('\n✓ Convergence plot saved to:', out_png)
    print('✓ Monitor CSV saved to:', monitor_file)


if __name__ == '__main__':
    main()
