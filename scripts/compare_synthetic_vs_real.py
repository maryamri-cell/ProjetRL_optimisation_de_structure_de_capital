"""
Compare résultats entraînement sur données synthétiques vs réelles
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os


def load_results(ticker, algorithm, data_type='real'):
    if data_type == 'real':
        path = f'results/real_data/{ticker}_{algorithm}_results.csv'
    else:
        path = f'results/synthetic/{algorithm}_results.csv'

    if not os.path.exists(path):
        print(f"⚠️  Results not found: {path}")
        return None

    return pd.read_csv(path)


def compare_distributions(real_results, synthetic_results, metric='rewards'):
    real_data = real_results[metric].values
    synth_data = synthetic_results[metric].values

    _, p_real = stats.shapiro(real_data[:min(50, len(real_data))])
    _, p_synth = stats.shapiro(synth_data[:min(50, len(synth_data))])

    if p_real > 0.05 and p_synth > 0.05:
        t_stat, p_value = stats.ttest_ind(real_data, synth_data)
        test_name = 't-test'
    else:
        u_stat, p_value = stats.mannwhitneyu(real_data, synth_data)
        test_name = 'Mann-Whitney U'

    mean_diff = np.mean(real_data) - np.mean(synth_data)
    pooled_std = np.sqrt((np.var(real_data) + np.var(synth_data)) / 2)
    cohens_d = mean_diff / (pooled_std + 1e-8)

    return {
        'test': test_name,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'mean_real': np.mean(real_data),
        'mean_synth': np.mean(synth_data),
        'std_real': np.std(real_data),
        'std_synth': np.std(synth_data),
        'improvement': (np.mean(real_data) - np.mean(synth_data)) / (abs(np.mean(synth_data)) + 1e-8) * 100
    }


def plot_comparison(real_results, synthetic_results, ticker, algorithm):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    metrics = ['rewards', 'lengths', 'leverages', 'waccs']
    titles = ['Reward', 'Episode Length', 'Final Leverage', 'WACC']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        ax.hist(synthetic_results[metric], bins=30, alpha=0.5, label='Synthetic', color='red', edgecolor='black')
        ax.hist(real_results[metric], bins=30, alpha=0.5, label='Real', color='green', edgecolor='black')
        ax.axvline(np.mean(synthetic_results[metric]), color='red', linestyle='--')
        ax.axvline(np.mean(real_results[metric]), color='green', linestyle='--')
        ax.set_xlabel(title)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title} Distribution')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f'{ticker} - {algorithm}: Synthetic vs Real Data Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_dir = 'results/comparisons'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/{ticker}_{algorithm}_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Comparison plot saved to {output_path}")


def main():
    ticker = 'AAPL'
    algorithm = 'PPO'

    print("="*80)
    print("SYNTHETIC VS REAL DATA COMPARISON")
    print("="*80)

    real_results = load_results(ticker, algorithm, 'real')
    synthetic_results = load_results(ticker, algorithm, 'synthetic')

    if real_results is None or synthetic_results is None:
        print("❌ Cannot proceed without both result files")
        return

    metrics = ['rewards', 'lengths', 'leverages']
    metric_names = ['Reward', 'Episode Length', 'Leverage']

    for metric, name in zip(metrics, metric_names):
        comp = compare_distributions(real_results, synthetic_results, metric)
        print(f"\n{name}:")
        print(f"  Real:      {comp['mean_real']:.2f} ± {comp['std_real']:.2f}")
        print(f"  Synthetic: {comp['mean_synth']:.2f} ± {comp['std_synth']:.2f}")
        print(f"  Test:      {comp['test']}")
        print(f"  P-value:   {comp['p_value']:.4f} {'✓ Significant' if comp['significant'] else '✗ Not significant'}")
        print(f"  Cohen's d: {comp['cohens_d']:.2f}")
        print(f"  Improvement: {comp['improvement']:+.1f}%")

    print(f"\n📊 Creating comparison plots...")
    plot_comparison(real_results, synthetic_results, ticker, algorithm)

    print("\n✓ Comparison complete!")


if __name__ == '__main__':
    main()
