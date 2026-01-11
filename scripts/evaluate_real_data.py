"""
Évalue modèles entraînés sur données réelles
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, SAC, TD3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config


def load_trained_model(ticker, algorithm, seed=42):
    model_path = f'models/real_data/{ticker}/{algorithm}_seed{seed}/best/best_model'
    if not os.path.exists(model_path + '.zip'):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if algorithm == 'PPO':
        model = PPO.load(model_path)
    elif algorithm == 'SAC':
        model = SAC.load(model_path)
    elif algorithm == 'TD3':
        model = TD3.load(model_path)

    return model


def evaluate_model(model, env, n_episodes=100):
    results = {
        'rewards': [],
        'lengths': [],
        'leverages': [],
        'coverages': [],
        'waccs': [],
        'enterprise_values': [],
        'defaults': []
    }

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

        results['rewards'].append(episode_reward)
        results['lengths'].append(step)
        results['leverages'].append(info.get('leverage', 0))
        results['coverages'].append(info.get('coverage', 0))
        results['waccs'].append(info.get('wacc', 0))
        results['enterprise_values'].append(info.get('enterprise_value', 0))
        results['defaults'].append(info.get('is_default', False))

    return results


def compute_statistics(results):
    stats = {
        'mean_reward': np.mean(results['rewards']),
        'std_reward': np.std(results['rewards']),
        'median_reward': np.median(results['rewards']),
        'mean_length': np.mean(results['lengths']),
        'success_rate': np.mean([l >= 240 for l in results['lengths']]),
        'default_rate': np.mean(results['defaults']),
        'mean_leverage': np.mean(results['leverages']),
        'std_leverage': np.std(results['leverages']),
        'mean_coverage': np.mean(np.clip(results['coverages'], 0, 100)),
        'mean_wacc': np.mean(results['waccs']),
        'mean_value': np.mean(results['enterprise_values'])
    }
    return stats


def plot_evaluation_results(results, ticker, algorithm, output_dir='results/real_data'):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].hist(results['rewards'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(results['rewards']), color='red', linestyle='--')
    axes[0, 0].set_title(f'{algorithm} - Reward Distribution')

    axes[0, 1].hist(results['lengths'], bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_title('Episode Length Distribution')

    axes[0, 2].hist(results['leverages'], bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[0, 2].axvline(0.4, color='red', linestyle='--')
    axes[0, 2].set_title('Leverage Distribution')

    coverages_clipped = np.clip(results['coverages'], 0, 20)
    axes[1, 0].hist(coverages_clipped, bins=30, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 0].set_title('Coverage Distribution')

    axes[1, 1].hist(results['waccs'], bins=30, alpha=0.7, edgecolor='black', color='red')
    axes[1, 1].set_title('WACC Distribution')

    metrics = [
        ('Success Rate', np.mean([l >= 240 for l in results['lengths']]) * 100),
        ('Default Rate', np.mean(results['defaults']) * 100),
        ('Avg Length', np.mean(results['lengths'])),
    ]

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    colors = ['green', 'red', 'blue']

    axes[1, 2].bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Success Metrics')

    plt.suptitle(f'{ticker} - {algorithm} Evaluation Results (n={len(results["rewards"])})', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = f'{output_dir}/{ticker}_{algorithm}_evaluation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC', 'TD3'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-episodes', type=int, default=100)

    args = parser.parse_args()

    print("="*80)
    print(f"EVALUATING {args.algorithm} ON {args.ticker}")
    print("="*80)

    model = load_trained_model(args.ticker, args.algorithm, args.seed)

    dataset = np.load('data/training/real_data_dataset.npy', allow_pickle=True).item()
    company_data = dataset[args.ticker]

    config = load_config('config.yaml')
    env = CapitalStructureEnv(
        config=config,
        real_cf_data=company_data['cf_normalized']
    )

    print(f"\n🧪 Evaluating on {args.n_episodes} episodes...")
    results = evaluate_model(model, env, args.n_episodes)

    stats = compute_statistics(results)

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"  Mean Reward:      {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Success Rate:     {stats['success_rate']*100:.1f}%")
    print(f"  Default Rate:     {stats['default_rate']*100:.1f}%")

    print(f"\n📊 Creating visualizations...")
    plot_evaluation_results(results, args.ticker, args.algorithm)

    results_df = pd.DataFrame(results)
    output_path = f'results/real_data/{args.ticker}_{args.algorithm}_results.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"💾 Results saved to {output_path}")

    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
