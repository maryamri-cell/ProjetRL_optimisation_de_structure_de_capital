"""
Entraînement PPO/SAC/TD3 sur données réelles
"""
import numpy as np
import torch
import argparse
import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import sys
import os

# Ajouter parent directory au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config
from src.utils.prioritized_replay_buffer import PrioritizedReplayBuffer
from scripts.callbacks import LearningRateScheduler, DetailedMonitoringCallback, StabilityMonitoringCallback


def load_real_data(ticker=None):
    dataset_path = 'data/training/real_data_dataset.npy'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Real data dataset not found at {dataset_path}\n"
            "Please run: python scripts/collect_real_data.py && python scripts/prepare_training_data.py"
        )

    dataset = np.load(dataset_path, allow_pickle=True).item()

    if ticker:
        if ticker not in dataset:
            raise ValueError(f"Ticker {ticker} not found in dataset. Available: {list(dataset.keys())}")
        return {ticker: dataset[ticker]}, [ticker]

    return dataset, list(dataset.keys())


def make_env(config, real_cf_data, seed, no_reward_normalization=False):
    def _init():
        env = CapitalStructureEnv(
            config=config,
            max_steps=min(252, len(real_cf_data)),
            scenario='baseline',
            real_cf_data=real_cf_data,
            disable_reward_normalization=no_reward_normalization
        )
        env = Monitor(env)
        try:
            env.reset(seed=seed)
        except Exception:
            env.reset()
        return env

    return _init


def train_single_company(
    ticker,
    algorithm='PPO',
    total_timesteps=2_000_000,
    n_envs=4,
    seed=42,
    no_reward_normalization=False
):
    print("="*80)
    print(f"TRAINING {algorithm} ON REAL DATA: {ticker}")
    print("="*80)

    dataset, _ = load_real_data(ticker=ticker)
    company_data = dataset[ticker]

    print(f"\n📊 Company Data:")
    print(f"   Ticker: {ticker}")
    print(f"   Sector: {company_data['sector']}")
    print(f"   Quarters: {company_data['n_quarters']}")
    print(f"   Period: {company_data['dates'][0]} to {company_data['dates'][-1]}")

    config = load_config('config.yaml')
    hyperparams_path = 'configs/optimized_hyperparams.yaml'

    with open(hyperparams_path, 'r') as f:
        hyperparams = yaml.safe_load(f)

    algo_params = hyperparams.get(algorithm, {})

    print(f"\n🏗️  Creating {n_envs} parallel environments...")

    envs = SubprocVecEnv([
        make_env(config, company_data['cf_normalized'], seed + i, no_reward_normalization)
        for i in range(n_envs)
    ])

    eval_env = DummyVecEnv([
        make_env(config, company_data['cf_normalized'], seed + 1000, no_reward_normalization)
    ])

    save_path = f'models/real_data/{ticker}/{algorithm}_seed{seed}'
    os.makedirs(save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{save_path}/best/',
        log_path=f'logs/real_data/{ticker}/{algorithm}_seed{seed}/',
        eval_freq=10_000,
        deterministic=True,
        n_eval_episodes=20,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=f'{save_path}/checkpoints/',
        name_prefix=f'{algorithm}_model'
    )

    lr_scheduler = LearningRateScheduler(
        initial_lr=algo_params.get('learning_rate', 3e-4),
        verbose=1
    )

    monitoring = DetailedMonitoringCallback(log_freq=1000)
    stability = StabilityMonitoringCallback(check_freq=10_000, verbose=1)

    callbacks = [eval_callback, checkpoint_callback, lr_scheduler, monitoring, stability]

    print(f"\n🤖 Initializing {algorithm} model...")
    print(f"   Hyperparameters: {algo_params}")

    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            envs,
            learning_rate=algo_params.get('learning_rate', 3e-4),
            n_steps=algo_params.get('n_steps', 2048),
            batch_size=algo_params.get('batch_size', 64),
            n_epochs=algo_params.get('n_epochs', 10),
            gamma=algo_params.get('gamma', 0.99),
            gae_lambda=algo_params.get('gae_lambda', 0.95),
            clip_range=algo_params.get('clip_range', 0.2),
            ent_coef=algo_params.get('ent_coef', 0.01),
            vf_coef=algo_params.get('vf_coef', 0.5),
            max_grad_norm=algo_params.get('max_grad_norm', 0.5),
            verbose=1,
            tensorboard_log=f'logs/real_data/{ticker}/tensorboard/'
        )

    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            envs,
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
            tensorboard_log=f'logs/real_data/{ticker}/tensorboard/'
        )
        # Remplacer le replay buffer standard par un prioritized buffer
        try:
            model.replay_buffer = PrioritizedReplayBuffer(
                buffer_size=algo_params.get('buffer_size', 100_000),
                observation_space=model.observation_space,
                action_space=model.action_space,
                device=model.device,
                n_envs=1,
                alpha=0.6,  # Prioritization strength
                beta=0.4    # Importance sampling strength
            )
            print(f"   ✓ Prioritized Replay Buffer enabled (alpha=0.6, beta=0.4)")
        except Exception as e:
            print(f"   ⚠ Could not enable prioritized replay buffer: {e}")
            print(f"   Using standard replay buffer instead")

    elif algorithm == 'TD3':
        model = TD3(
            'MlpPolicy',
            envs,
            learning_rate=algo_params.get('learning_rate', 3e-4),
            buffer_size=algo_params.get('buffer_size', 1_000_000),
            learning_starts=algo_params.get('learning_starts', 10_000),
            batch_size=algo_params.get('batch_size', 256),
            tau=algo_params.get('tau', 0.005),
            gamma=algo_params.get('gamma', 0.99),
            policy_delay=algo_params.get('policy_delay', 2),
            target_policy_noise=algo_params.get('target_policy_noise', 0.2),
            verbose=1,
            tensorboard_log=f'logs/real_data/{ticker}/tensorboard/'
        )

    print(f"\n🚀 Starting training...")
    print(f"   Total timesteps: {total_timesteps:,}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    final_path = f'{save_path}/final_model'
    model.save(final_path)

    print(f"\n✓ Training complete!")
    print(f"✓ Final model saved to: {final_path}")

    envs.close()
    eval_env.close()

    return model


def train_multi_company(
    tickers,
    algorithm='PPO',
    total_timesteps=5_000_000,
    seed=42,
    no_reward_normalization=False
):
    print("="*80)
    print(f"TRAINING {algorithm} ON MULTIPLE COMPANIES")
    print("="*80)

    dataset, all_tickers = load_real_data()

    if tickers:
        selected_tickers = [t for t in tickers if t in all_tickers]
    else:
        selected_tickers = all_tickers[:5]

    print(f"\n📊 Training on {len(selected_tickers)} companies:")
    for ticker in selected_tickers:
        data = dataset[ticker]
        print(f"   - {ticker} ({data['sector']}): {data['n_quarters']} quarters")

    config = load_config('config.yaml')
    hyperparams_path = 'configs/optimized_hyperparams.yaml'

    with open(hyperparams_path, 'r') as f:
        hyperparams = yaml.safe_load(f)

    algo_params = hyperparams.get(algorithm, {})

    print(f"\n🏗️  Creating {len(selected_tickers)} parallel environments (1 per company)...")

    envs = []
    for i, ticker in enumerate(selected_tickers):
        company_data = dataset[ticker]
        env_fn = make_env(config, company_data['cf_normalized'], seed + i, no_reward_normalization)
        envs.append(env_fn)

    vec_env = SubprocVecEnv(envs)

    eval_envs = [
        make_env(config, dataset[ticker]['cf_normalized'], seed + 1000 + i, no_reward_normalization)
        for i, ticker in enumerate(selected_tickers[:2])
    ]
    eval_env = DummyVecEnv(eval_envs)

    save_path = f'models/real_data/multi_company/{algorithm}_seed{seed}'
    os.makedirs(save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{save_path}/best/',
        log_path=f'logs/real_data/multi_company/{algorithm}_seed{seed}/',
        eval_freq=20_000,
        deterministic=True,
        n_eval_episodes=10,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=f'{save_path}/checkpoints/',
        name_prefix=f'{algorithm}_multi'
    )

    lr_scheduler = LearningRateScheduler(algo_params.get('learning_rate', 3e-4), verbose=1)
    monitoring = DetailedMonitoringCallback(log_freq=1000)

    callbacks = [eval_callback, checkpoint_callback, lr_scheduler, monitoring]

    print(f"\n🤖 Initializing {algorithm} model for multi-company training...")

    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            vec_env,
            **{k: v for k, v in algo_params.items()},
            verbose=1,
            tensorboard_log=f'logs/real_data/multi_company/tensorboard/'
        )
    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            vec_env,
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
            tensorboard_log=f'logs/real_data/multi_company/tensorboard/'
        )
        # Remplacer le replay buffer standard par un prioritized buffer
        try:
            model.replay_buffer = PrioritizedReplayBuffer(
                buffer_size=algo_params.get('buffer_size', 100_000),
                observation_space=model.observation_space,
                action_space=model.action_space,
                device=model.device,
                n_envs=len(selected_tickers),
                alpha=0.6,
                beta=0.4
            )
            print(f"   ✓ Prioritized Replay Buffer enabled (alpha=0.6, beta=0.4)")
        except Exception as e:
            print(f"   ⚠ Could not enable prioritized replay buffer: {e}")
            print(f"   Using standard replay buffer instead")
    elif algorithm == 'TD3':
        model = TD3(
            'MlpPolicy',
            vec_env,
            **{k: v for k, v in algo_params.items()},
            verbose=1,
            tensorboard_log=f'logs/real_data/multi_company/tensorboard/'
        )

    print(f"\n🚀 Starting multi-company training...")
    print(f"   Total timesteps: {total_timesteps:,}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    final_path = f'{save_path}/final_model'
    model.save(final_path)

    vec_env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description='Train RL agent on real company data')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multi'],
                      help='Training mode: single company or multi-company')
    parser.add_argument('--ticker', type=str, default='AAPL',
                      help='Company ticker for single mode (default: AAPL)')
    parser.add_argument('--tickers', type=str, nargs='+', default=None,
                      help='List of tickers for multi mode')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC', 'TD3'],
                      help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=2_000_000,
                      help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=4,
                      help='Number of parallel environments (single mode)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--no-reward-normalization', action='store_true',
                      help='Disable running reward normalization (use raw rewards)')

    args = parser.parse_args()
    
    # Pass normalization flag to environment via config
    if args.no_reward_normalization:
        print("\n⚠️  WARNING: Reward normalization DISABLED - using raw rewards only")
    else:
        print("\n✓ Reward normalization ENABLED - using fixed clipping")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == 'single':
        model = train_single_company(
            ticker=args.ticker,
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            no_reward_normalization=args.no_reward_normalization
        )
    else:
        model = train_multi_company(
            tickers=args.tickers,
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            seed=args.seed,
            no_reward_normalization=args.no_reward_normalization
        )

    print("\n" + "="*80)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == '__main__':
    main()
