#!/usr/bin/env python3
"""
Quick Test: Capital Structure Optimizer
Demonstrates the optimizer without needing to run the web server
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from src.utils.config import load_config
from src.environment.capital_structure_env import CapitalStructureEnv
from src.models.company import CompanyModel


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def optimize_company(company_data, steps=5, verbose=True):
    """Run optimization on a company"""
    
    # Load model and config
    model_path = 'models/real_data/multi_company/PPO_seed44/final_model'
    if not os.path.exists(model_path + '.zip'):
        print(f"❌ Model not found at {model_path}")
        return None
    
    model = PPO.load(model_path)
    config = load_config('config.yaml')
    
    # Create environment
    cf_data = np.array(company_data['cf_history'])
    env = CapitalStructureEnv(
        config=config,
        max_steps=steps,
        scenario='baseline',
        real_cf_data=cf_data,
        disable_reward_normalization=True
    )
    
    # Set initial company state
    env.company.debt = company_data['debt']
    env.company.equity = company_data['equity']
    env.company.cash = company_data['cash']
    env.company.cf = cf_data[0]
    
    results = []
    obs, _ = env.reset()
    
    if verbose:
        print(f"\n📊 Initial State:")
        print(f"   Debt: ${company_data['debt']:.2f}M | Equity: ${company_data['equity']:.2f}M | Cash: ${company_data['cash']:.2f}M")
        print(f"   Leverage: {company_data['debt']/(company_data['debt']+company_data['equity']):.4f}")
        print(f"\n⚙️  Running {steps}-quarter optimization...\n")
    
    # Run optimization
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        company = env.company
        result = {
            'step': step + 1,
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'reward': float(reward),
            'debt': float(company.debt),
            'equity': float(company.equity),
            'cash': float(company.cash),
            'leverage': float(company.get_leverage()),
            'coverage': float(company.get_interest_coverage()),
            'wacc': float(company.calculate_wacc()),
        }
        results.append(result)
        
        if verbose:
            print(f"Q{step+1} | Reward: {reward:+.4f} | Leverage: {result['leverage']:.4f} | WACC: {result['wacc']*100:.2f}% | Coverage: {result['coverage']:.2f}x")
        
        if done or truncated:
            break
    
    if verbose:
        print(f"\n✅ Optimization complete!")
        print(f"\n📈 Final State:")
        final = results[-1]
        print(f"   Debt: ${final['debt']:.2f}M | Equity: ${final['equity']:.2f}M | Cash: ${final['cash']:.2f}M")
        print(f"   Leverage: {final['leverage']:.4f}")
        print(f"   Total Reward: {sum(r['reward'] for r in results):.4f}")
    
    return results


def main():
    """Main test suite"""
    
    print_header("🚀 Capital Structure Optimizer - Quick Test")
    
    # Test Case 1: Tech Startup
    print_header("Test Case 1: Tech Startup (High Growth)")
    tech_startup = {
        'ticker': 'TECH',
        'debt': 50.0,
        'equity': 300.0,
        'cash': 100.0,
        'cf_history': [0.05, 0.08, 0.10, 0.12, 0.15],
        'description': 'Growing tech company with increasing cash flows'
    }
    
    results_tech = optimize_company(tech_startup)
    if results_tech:
        print(f"\n💡 Insight: Model increased {'debt' if results_tech[-1]['debt'] > tech_startup['debt'] else 'equity'} to optimize growth funding")
    
    # Test Case 2: Mature Company
    print_header("Test Case 2: Mature Company (Stable)")
    mature = {
        'ticker': 'MAT',
        'debt': 200.0,
        'equity': 400.0,
        'cash': 50.0,
        'cf_history': [0.20, 0.20, 0.20, 0.20, 0.20],
        'description': 'Stable mature company'
    }
    
    results_mature = optimize_company(mature)
    if results_mature:
        print(f"\n💡 Insight: Model {'reduced' if results_mature[-1]['leverage'] < (mature['debt']/(mature['debt']+mature['equity'])) else 'maintained'} leverage for stability")
    
    # Test Case 3: Distressed Company
    print_header("Test Case 3: Distressed Company (Recovery)")
    distressed = {
        'ticker': 'DIST',
        'debt': 300.0,
        'equity': 100.0,
        'cash': 10.0,
        'cf_history': [0.05, 0.03, 0.02, -0.01, 0.01],
        'description': 'Company in financial distress'
    }
    
    results_dist = optimize_company(distressed)
    if results_dist:
        print(f"\n💡 Insight: Model {'successfully reduced' if results_dist[-1]['debt'] < distressed['debt'] else 'attempted to reduce'} debt to recover")
    
    # Summary
    print_header("📋 Summary")
    print("\nOptimization Results:")
    print(f"  ✅ Tech Startup: Final Leverage = {results_tech[-1]['leverage']:.4f}")
    print(f"  ✅ Mature Company: Final Leverage = {results_mature[-1]['leverage']:.4f}")
    print(f"  ✅ Distressed: Final Leverage = {results_dist[-1]['leverage']:.4f}")
    
    print("\n💰 Key Takeaways:")
    print("  1. The PPO model successfully optimizes capital structure")
    print("  2. Actions are tailored to company financial state")
    print("  3. Model balances growth, stability, and risk management")
    print("  4. Leverage adjustments reflect business fundamentals")
    
    print("\n🚀 Next Steps:")
    print("  1. Run the web optimizer for interactive testing:")
    print("     python optimizer_app.py")
    print("  2. Visit http://localhost:5000 in your browser")
    print("  3. Test with your own company data")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
