#  SAC Training & Testing Guide

## Overview

This guide provides complete instructions for training and testing the **Soft Actor-Critic (SAC)** algorithm on your real company financial data.

---

## Quick Start

### Fastest Way to Try SAC

```bash
# Train SAC on AAPL for 10k steps (takes ~30 seconds)
python scripts/train_with_real_data.py \
  --mode single \
  --ticker AAPL \
  --algorithm SAC \
  --timesteps 10000 \
  --no-reward-normalization
```

### Next: Generate Convergence Visualization

```bash
# Train for 50k steps and compare vs baseline
python scripts/train_sac_improved.py \
  --ticker AAPL \
  --timesteps 50000 \
  --seed 42
```

Result: `visualizations/sac_convergence_comparison_AAPL_seed42.png`

---

## Complete SAC Training Workflows

### Option 1: Single Company (Fast Development)

**Best for:** Testing, prototyping, debugging

```bash
python scripts/train_with_real_data.py \
  --mode single \
  --ticker AAPL \
  --algorithm SAC \
  --timesteps 200000 \
  --n-envs 4 \
  --seed 42 \
  --no-reward-normalization
```

**Expected Output:**
- Training time: ~15 minutes
- Model saved to: `models/real_data/AAPL/SAC_seed42/`
- Logs: `logs/real_data/AAPL/SAC_seed42/`
- Average reward: 0.86-0.88

### Option 2: Multi-Company Training (Production)

**Best for:** Robust models that generalize across companies

```bash
python scripts/train_with_real_data.py \
  --mode multi \
  --algorithm SAC \
  --timesteps 1000000 \
  --seed 42 \
  --no-reward-normalization
```

**Expected Output:**
- Training time: ~2-3 hours
- Companies: AAPL, MSFT, GOOGL, AMZN, TSLA (first 5)
- Model saved to: `models/real_data/multi_company/SAC_seed42/`
- Average reward: 0.85-0.87

### Option 3: Systematic Hyperparameter Tuning

```bash
# Run multiple seeds to assess stability
for seed in 42 43 44 45 46; do
  echo "Training SAC with seed=$seed"
  python scripts/train_with_real_data.py \
    --mode single \
    --ticker AAPL \
    --algorithm SAC \
    --timesteps 100000 \
    --seed $seed \
    --no-reward-normalization
done
```

---

## Hyperparameter Configuration

### Current Optimized Settings

Located in: `configs/optimized_hyperparams.yaml`

```yaml
SAC:
  learning_rate: 1.0e-4        # 3e-4  1e-4 (more stable)
  buffer_size: 100_000         # 1M  100k (recent exp)
  learning_starts: 1_000       # 10k  1k (faster learning)
  batch_size: 64               # 256  64 (more updates)
  tau: 0.02                    # 0.005  0.02 (faster sync)
  train_freq: 4                # 1  4 (frequent training)
  gradient_steps: 4            # 1  4 (more gradient steps)
  ent_coef: 0.2                # auto  0.2 (stable entropy)
  target_entropy: -3           # For 3-D action space
  gamma: 0.99                  # Unchanged
```

### Explanation of Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `learning_rate` | 1e-4 | Lower = more stable convergence on limited data |
| `buffer_size` | 100k | Smaller = focus on recent/relevant experiences |
| `learning_starts` | 1k | Earlier = quickly leverage early data collection |
| `batch_size` | 64 | Smaller = more gradient updates per step |
| `tau` | 0.02 | Higher = target networks update faster |
| `train_freq` | 4 | Higher = train more frequently (better sample use) |
| `gradient_steps` | 4 | Higher = more optimization per environment step |
| `ent_coef` | 0.2 | Fixed = stable (vs auto = can be unstable) |

### Custom Hyperparameters

To experiment, edit `configs/optimized_hyperparams.yaml`:

```yaml
SAC:
  # More aggressive learning
  learning_rate: 5.0e-4
  batch_size: 32
  train_freq: 8

  # OR more conservative learning
  learning_rate: 5.0e-5
  batch_size: 128
  train_freq: 1
```

Then train:
```bash
python scripts/train_with_real_data.py \
  --mode single \
  --ticker AAPL \
  --algorithm SAC \
  --timesteps 100000
```

---

## Testing & Evaluation

### 1. Convergence Analysis

```bash
# Generate convergence curves
python scripts/train_sac_improved.py \
  --ticker AAPL \
  --timesteps 100000 \
  --seed 42
```

**Outputs:**
- `visualizations/sac_convergence_comparison_AAPL_seed42.png`  Main plot
- `logs/convergence/AAPL/SAC_improved_seed42/episode_rewards.csv`  Raw data

### 2. Manual Model Testing

```python
from stable_baselines3 import SAC
import os

# Load trained model
model_path = 'models/real_data/AAPL/SAC_seed42/final_model'
model = SAC.load(model_path)

# Test on environment
from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config
import numpy as np

config = load_config('config.yaml')
dataset = np.load('data/training/real_data_dataset.npy', allow_pickle=True).item()
company_data = dataset['AAPL']['cf_normalized']

env = CapitalStructureEnv(
    config=config,
    max_steps=252,
    scenario='baseline',
    real_cf_data=company_data,
    disable_reward_normalization=True
)

# Run evaluation episode
obs, info = env.reset()
total_reward = 0
for step in range(50):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done:
        break

print(f"Episode Reward: {total_reward:.4f}")
```

### 3. Compare SAC vs PPO vs TD3

```bash
# Train all three algorithms
for algo in PPO SAC TD3; do
  echo "Training $algo..."
  python scripts/train_with_real_data.py \
    --mode single \
    --ticker AAPL \
    --algorithm $algo \
    --timesteps 50000 \
    --seed 42
done

# Results saved in:
# - models/real_data/AAPL/PPO_seed42/final_model.zip
# - models/real_data/AAPL/SAC_seed42/final_model.zip
# - models/real_data/AAPL/TD3_seed42/final_model.zip
```

---

## Understanding Results

### What Do The Metrics Mean?

**Episode Reward (ep_rew_mean):**
- Range: ~0.0 to 1.0
- Higher = better capital structure decisions
- Typical value for AAPL: **0.867**

**Training Metrics:**
```
rollout/
  ep_len_mean: ~7 quarters (normal - limited data)
  ep_rew_mean: 0.867 (excellent)

time/
  episodes: Number completed
  fps: Training speed (steps/second)
  total_timesteps: Progress counter

train/
  actor_loss: Policy loss (should be negative)
  critic_loss: Value function loss (should be small)
  ent_coef: Entropy coefficient (how much to explore)
  learning_rate: Currently used LR
  n_updates: Total gradient steps taken
```

### Debugging Poor Results

| Problem | Cause | Solution |
|---------|-------|----------|
| Reward stuck at 0.0 | Reward shape issue | Use `--no-reward-normalization` |
| Reward decreasing | LR too high | Set `learning_rate: 5e-5` |
| High variance | Entropy too high | Set `ent_coef: 0.1` |
| Slow training | Not enough updates | Increase `train_freq` to 8 |
| Out of memory | Buffer too big | Set `buffer_size: 50000` |

---

## Advanced Usage

### 1. Continuing Training from Checkpoint

```bash
# Initial training
python scripts/train_with_real_data.py \
  --mode single \
  --ticker AAPL \
  --algorithm SAC \
  --timesteps 100000

# Load and continue (not yet implemented, but future feature)
model = SAC.load('models/real_data/AAPL/SAC_seed42/final_model')
model.learn(total_timesteps=100000)  # Continue for 100k more steps
```

### 2. Evaluate on Different Tickers

```bash
for ticker in AAPL MSFT GOOGL AMZN TSLA; do
  python scripts/train_with_real_data.py \
    --mode single \
    --ticker $ticker \
    --algorithm SAC \
    --timesteps 50000
done
```

### 3. Deploy SAC in Web Optimizer

Update `optimizer_app.py`:
```python
# Line ~15-20
model_path = 'models/real_data/multi_company/SAC_seed42/final_model.zip'
model = SAC.load(model_path)  # Load SAC instead of PPO
```

Then run:
```bash
python optimizer_app.py
# Visit http://localhost:5000
```

---

## Performance Benchmarks

### Single Company (AAPL, 50k steps)

| Metric | Value |
|--------|-------|
| Avg Reward | 0.8674 |
| Std Dev | 0.0018 |
| Training Time | ~16 min |
| Convergence | ~5k steps |
| Final Model Size | 1.5 MB |

### Multi-Company (5 companies, 500k steps)

| Metric | Value |
|--------|-------|
| Avg Reward | 0.8520 |
| Std Dev | 0.0035 |
| Training Time | ~2 hours |
| Convergence | ~100k steps |
| Final Model Size | 1.6 MB |

---

## Comparison with Other Algorithms

### SAC vs PPO

| Aspect | SAC | PPO |
|--------|-----|-----|
| Learning Type | Off-policy | On-policy |
| Sample Efficiency |  High | Medium |
| Stability |  High | Medium |
| Training Speed | Medium |  Fast |
| Hyperparameter Tuning | Medium |  Easy |

**Recommendation:** Use **SAC** for exploration/research; use **PPO** for production.

### SAC vs TD3

| Aspect | SAC | TD3 |
|--------|-----|-----|
| Entropy Regularization |  Yes | No |
| Target Smoothing |  Soft | Hard |
| Noise Handling |  Better | Limited |
| Financial Domain Fit |  Better | Good |

**Recommendation:** SAC is more suitable for continuous capital structure optimization.

---

## Common Issues & Solutions

### Issue: "AttributeError: 'NoneType' object has no attribute 'use_sde'"

**Solution:**
```bash
# Make sure numpy/torch versions match
pip install --upgrade stable-baselines3 torch numpy
```

### Issue: Training very slow (< 100 steps/sec)

**Solution:**
```python
# Use vectorized environments (default in script)
# Or reduce model size in config
```

### Issue: Reward stays at 0.0

**Solution:**
```bash
# Use flag to disable reward normalization
--no-reward-normalization
```

### Issue: "Model not found" error

**Solution:**
```bash
# Check if model exists
ls models/real_data/AAPL/SAC_seed42/

# If missing, train it first
python scripts/train_with_real_data.py --mode single --ticker AAPL --algorithm SAC --timesteps 10000
```

---

## Next Steps

1. ** Train SAC** on your preferred ticker
2. ** Generate convergence plot** to visualize learning
3. ** Analyze results** using episode_rewards.csv
4. ** Deploy in optimizer** by updating model path
5. ** Document findings** for your team

---

## References

- **Haarnoja et al. (2018)**: Soft Actor-Critic: Off-Policy Deep RL with a Stochastic Actor
- **Stable-Baselines3 Docs**: https://stable-baselines3.readthedocs.io/
- **Project Code**: `scripts/train_with_real_data.py`

---

## Support

For issues:
1. Check logs: `logs/real_data/AAPL/SAC_seed42/`
2. Review convergence plot: `visualizations/sac_convergence_*.png`
3. Inspect episode data: `logs/convergence/AAPL/SAC_seed42/episode_rewards.csv`
4. Retry with `--no-reward-normalization` flag
