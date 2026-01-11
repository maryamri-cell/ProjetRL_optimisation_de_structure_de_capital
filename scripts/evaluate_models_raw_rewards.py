"""Evaluate saved SAC/TD3 models with reward normalization disabled and save raw returns."""
import os
import numpy as np
import csv
import argparse
import sys

from stable_baselines3 import SAC, TD3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config


def evaluate_model(model, env, episodes=200):
    returns = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out
            # reward here is normalized or raw depending on env flag
            total += float(reward)
        returns.append(total)
    return returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--models', nargs='+', default=['TD3','SAC'])
    args = parser.parse_args()

    data = np.load('data/training/real_data_dataset.npy', allow_pickle=True).item()
    company = data[args.ticker]
    company_cf = company['cf_normalized']

    config = load_config('config.yaml')

    # create env with reward normalization disabled
    env = CapitalStructureEnv(config=config, max_steps=min(252, len(company_cf)), scenario='baseline', real_cf_data=company_cf, disable_reward_normalization=True)

    for algo in args.models:
        model_path = os.path.join('models', 'real_data', args.ticker.lower(), f'{algo}_seed42', 'final_model')
        model_zip = model_path + '.zip'
        if os.path.exists(model_zip):
            load_path = model_zip
        elif os.path.exists(model_path):
            load_path = model_path
        else:
            print(f'Model for {algo} not found at {model_path}(.zip)')
            continue

        print(f'Loading {algo} from {load_path}')
        try:
            if algo == 'SAC':
                model = SAC.load(load_path)
            elif algo == 'TD3':
                model = TD3.load(load_path)
            else:
                print(f'Unsupported algo: {algo}')
                continue
        except Exception as e:
            print('Failed to load model:', e)
            continue

        returns = evaluate_model(model, env, episodes=args.episodes)
        out_dir = os.path.join('logs', 'evaluation_raw', args.ticker)
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f'{algo}_raw_returns.csv')
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'return'])
            for i,r in enumerate(returns, 1):
                writer.writerow([i, r])
        print(f'{algo} evaluated: mean={np.mean(returns):.6f} std={np.std(returns):.6f} saved to {out_csv}')

if __name__ == '__main__':
    main()
