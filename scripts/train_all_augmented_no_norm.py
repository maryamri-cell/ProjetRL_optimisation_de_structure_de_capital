"""Train PPO, SAC, TD3 on augmented real data without reward normalization.

Usage:
    python scripts/train_all_augmented_no_norm.py --ticker AAPL --timesteps 100000 --seeds 1
"""
import os
import sys
import argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import csv

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config


def load_augmented_dataset():
    path = os.path.join('data', 'training', 'real_data_dataset_augmented_20.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True).item()


def make_env_factory(config, company_data, seed, log_path=None):
    def _init():
        env = CapitalStructureEnv(
            config=config,
            max_steps=min(252, len(company_data)),
            scenario='baseline',
            real_cf_data=company_data,
            disable_reward_normalization=True
        )
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            env = Monitor(env, filename=log_path)
        try:
            env.reset(seed=seed)
        except Exception:
            env.reset()
        return env
    return _init


def train_one_algo(algo_name, company_cf, config, timesteps, seed, n_envs=1):
    print(f"\nTraining {algo_name} | ticker CF shape: {company_cf.shape} | timesteps: {timesteps} | seed: {seed}")

    log_dir = os.path.join('logs', 'convergence', 'AUGMENTED', f'{algo_name}_aug_no_norm_seed{seed}')
    os.makedirs(log_dir, exist_ok=True)
    monitor_file = os.path.join(log_dir, 'monitor.csv')

    env_fn = make_env_factory(config, company_cf, seed, monitor_file)
    env = SubprocVecEnv([env_fn for _ in range(n_envs)]) if n_envs > 1 else SubprocVecEnv([env_fn])

    # default hyperparams
    if algo_name == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1)
    elif algo_name == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1)
    elif algo_name == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1)
    else:
        raise ValueError('Unknown algorithm')

    model.learn(total_timesteps=timesteps)

    model_path = os.path.join('models', 'real_data', 'augmented', f'{algo_name}_aug_no_norm_seed{seed}')
    os.makedirs(model_path, exist_ok=True)
    model.save(os.path.join(model_path, 'final_model'))
    env.close()

    # parse monitor.csv
    rewards = []
    try:
        with open(monitor_file, 'r') as f:
            lines = f.readlines()
            start_idx = 1 if lines and lines[0].startswith('#') else 0
            reader = csv.DictReader(lines[start_idx:])
            for row in reader:
                try:
                    rewards.append(float(row.get('r', row.get('reward', 0))))
                except Exception:
                    pass
    except FileNotFoundError:
        print('Warning: monitor file not found:', monitor_file)

    out_csv = os.path.join(log_dir, 'episode_rewards.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward'])
        for i, r in enumerate(rewards, 1):
            writer.writerow([i, r])

    print(f"Saved model to {model_path}, monitor to {monitor_file}, episode CSV to {out_csv}")
    return rewards, log_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--n-envs', type=int, default=1)
    parser.add_argument('--algos', nargs='+', default=['PPO','SAC','TD3'])
    args = parser.parse_args()

    dataset = load_augmented_dataset()
    # tolerant ticker lookup (case-insensitive)
    key = None
    for k in dataset.keys():
        if k.lower() == args.ticker.lower():
            key = k
            break
    if key is None:
        raise KeyError(f"Ticker {args.ticker} not found in augmented dataset")

    company = dataset[key]
    company_cf = company.get('cf_normalized', company.get('cf_raw', None))
    # If augmented variants present (2D), flatten to 1D sequence for env compatibility
    if hasattr(company_cf, 'ndim') and company_cf.ndim == 2:
        # flatten variants into a single long series
        company_cf = company_cf.flatten()
    if company_cf is None:
        raise ValueError('No cf data found for ticker')

    config = load_config('config.yaml')

    for algo in args.algos:
        for seed in range(args.seeds):
            s = seed + 1
            train_one_algo(algo, company_cf, config, args.timesteps, s, n_envs=args.n_envs)


if __name__ == '__main__':
    main()
