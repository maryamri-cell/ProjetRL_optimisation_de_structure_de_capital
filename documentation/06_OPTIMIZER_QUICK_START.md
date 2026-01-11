#  Capital Structure Optimizer - Quick Start Guide

## Overview

The **Capital Structure Optimizer** is a web-based interface for testing your trained PPO model. It allows you to:

-  **Input company financial data** (debt, equity, cash flows)
-  **Run AI optimization** using the trained PPO model
-  **View results** with detailed metrics and improvements
-  **Compare strategies** (AI vs baseline approaches)

---

## Installation & Setup

### 1. Install Required Dependencies

```bash
pip install flask flask-cors stable-baselines3 gymnasium
```

### 2. Verify Model File Exists

The optimizer requires the trained model at:
```
models/real_data/multi_company/PPO_seed44/final_model.zip
```

Check if it exists:
```bash
# Windows
dir models\real_data\multi_company\PPO_seed44\

# Mac/Linux
ls models/real_data/multi_company/PPO_seed44/
```

---

## Running the Optimizer

### Option A: Web Interface (Recommended)

#### Windows:
```bash
python optimizer_app.py
```

#### Mac/Linux:
```bash
python3 optimizer_app.py
```

**Output:**
```
======================================================================
   Capital Structure Optimizer
======================================================================
  Model Status:  LOADED
  Web Interface: http://localhost:5000
======================================================================
```

**Access the interface:**
- Open browser to: `http://localhost:5000`
- Fill in company data
- Click "Optimize"
- View results in real-time

### Option B: Command-Line Testing

```bash
python test_optimizer.py
```

This runs 3 pre-configured test cases:
1. **Tech Startup** - Growing company with increasing cash flows
2. **Mature Company** - Stable business with consistent cash flows
3. **Distressed Company** - In recovery with negative cash flows

**Example Output:**
```
======================================================================
   Capital Structure Optimizer - Quick Test
======================================================================

Test Case 1: Tech Startup (High Growth)
======================================================================

 Initial State:
   Debt: $50.00M | Equity: $300.00M | Cash: $100.00M
   Leverage: 0.1429

  Running 5-quarter optimization...

Q1 | Reward: +0.2340 | Leverage: 0.1425 | WACC: 8.15% | Coverage: 2.45x
Q2 | Reward: +0.2145 | Leverage: 0.1430 | WACC: 8.18% | Coverage: 2.42x
Q3 | Reward: +0.1998 | Leverage: 0.1438 | WACC: 8.22% | Coverage: 2.38x
Q4 | Reward: +0.1875 | Leverage: 0.1445 | Leverage: 0.1445 | Coverage: 2.35x
Q5 | Reward: +0.1762 | Leverage: 0.1450 | WACC: 8.28% | Coverage: 2.32x

 Optimization complete!

 Final State:
   Debt: $52.15M | Equity: $310.42M | Cash: $102.35M
   Leverage: 0.1450
   Total Reward: 1.0120
```

---

## Web Interface Features

###  Input Panel

**Company Data Section:**
- **Ticker Symbol**: Company identifier (e.g., "ACME")
- **Total Debt**: Current debt in millions (e.g., 100)
- **Total Equity**: Current equity in millions (e.g., 200)
- **Cash & Equivalents**: Available cash (e.g., 50)
- **Cash Flow History**: 5 quarters of past cash flows
  - Format: `0.10, 0.12, 0.15, 0.18, 0.20`
- **Optimization Horizon**: How many quarters to optimize (1-10)

###  Quick Load Examples

Click one of these buttons to auto-fill example data:

1. **Tech Startup**
   - Scenario: Growing company
   - Debt: $50M | Equity: $300M
   - CF Trend: Increasing (0.05  0.15)

2. **Mature Company**
   - Scenario: Stable business
   - Debt: $200M | Equity: $400M
   - CF Trend: Constant (0.20)

3. **Distressed**
   - Scenario: In recovery
   - Debt: $300M | Equity: $100M
   - CF Trend: Volatile with negative (-0.01)

4. **Growth Company**
   - Scenario: High growth, reinvesting
   - Debt: $100M | Equity: $250M
   - CF Trend: Strong growth (0.10  0.30)

###  Results Panel

**Summary Metrics:**
- **Total Reward**: Sum of rewards across all quarters (higher is better)
- **Avg Reward/Step**: Average reward per quarter
- **Leverage Change**: How debt-to-capital ratio changed

**Quarterly Timeline:**
- Shows quarter-by-quarter optimization
- Displays debt, equity, leverage, and reward
- Visualizes how capital structure evolved

**Detailed Results Table:**
- Complete metrics for each quarter
- Columns: Debt, Equity, Leverage, Coverage, WACC
- Coverage: Interest coverage ratio (higher is safer)
- WACC: Weighted average cost of capital (lower is better)

---

## Understanding Results

###  Key Metrics Explained

**Leverage (Debt-to-Capital Ratio)**
- Formula: `Debt / (Debt + Equity)`
- Range: 0% (all equity) to 100% (all debt)
- Interpretation:
  -  Lower = Less risky, higher interest coverage
  -  Higher = More risky, but potentially higher returns

**Interest Coverage**
- Formula: `EBIT / Interest Expense`
- Interpretation:
  - > 2.0x = Comfortable debt level
  - 1.0-2.0x = Moderate risk
  - < 1.0x = Distressed (can't cover interest)

**WACC (Weighted Average Cost of Capital)**
- Interpretation:
  - Lower = More value created
  - Higher = Riskier capital structure
  - Optimal around 7-9% for most companies

**Reward Signal**
- Represents how good each action was
- Ranges: -1.0 to +1.0
- Higher values = Better capital structure decision

###  What Good Results Look Like

**Tech Startup:**
-  Rewards increasing: Company learning to use cheap debt
-  Leverage increasing slightly: Financing growth
-  Coverage staying > 2.0x: Maintaining safety

**Mature Company:**
-  Stable rewards: Balanced capital structure
-  Consistent leverage: No major changes needed
-  WACC stable: Optimized already

**Distressed Company:**
-  Rewards initially negative then improving: Recovery process
-  Leverage decreasing: Reducing debt burden
-  Coverage improving: Moving toward safety

---

## Testing Scenarios

### Scenario 1: Can the model fund growth?

**Setup:**
- Ticker: `GROW`
- Debt: 50M, Equity: 200M, Cash: 100M
- CF: `0.15, 0.17, 0.20, 0.22, 0.25`
- Steps: 5

**Expected Result:**
- Model should increase debt modestly to fund growth
- Coverage stays > 2.0x
- WACC drops (cheaper financing)

### Scenario 2: Can the model handle distress?

**Setup:**
- Ticker: `DIST`
- Debt: 300M, Equity: 100M, Cash: 10M
- CF: `0.10, 0.08, 0.05, 0.03, 0.01` (declining)
- Steps: 5

**Expected Result:**
- Model reduces debt aggressively
- Coverage improves (avoid bankruptcy)
- Leverage drops significantly

### Scenario 3: Is the model stable?

**Setup:**
- Same company data, run 2x with different horizons
- First: 5 steps, Second: 10 steps

**Expected Result:**
- Decisions are consistent
- First 5 steps match in both runs
- No erratic behavior

---

## API Endpoints

If you want to integrate the optimizer into your own application:

### GET `/api/model-status`
Check if model is loaded

**Response:**
```json
{
  "loaded": true,
  "error": null,
  "model_path": "models/real_data/multi_company/PPO_seed44/final_model.zip"
}
```

### POST `/api/optimize`
Run optimization on company data

**Request Body:**
```json
{
  "ticker": "ACME",
  "debt": 100,
  "equity": 200,
  "cash": 50,
  "cf_history": [0.10, 0.12, 0.15, 0.18, 0.20],
  "steps": 5
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "step": 1,
      "action": 0.1234,
      "reward": 0.2340,
      "debt": 100.50,
      "equity": 200.25,
      "leverage": 0.3338,
      "coverage": 2.45,
      "wacc": 8.15
    },
    ...
  ],
  "total_reward": 1.0120,
  "avg_reward": 0.2024,
  "improvement": {
    "leverage_change": 0.0038,
    "wacc_change": -0.15
  }
}
```

### GET `/api/example-companies`
Get example company data

**Response:**
```json
{
  "tech_startup": {
    "name": "Tech Startup",
    "ticker": "TECH",
    "debt": 50,
    "equity": 300,
    "cash": 100,
    "cf_history": [0.05, 0.08, 0.10, 0.12, 0.15],
    "description": "Growing tech company with increasing cash flows"
  },
  ...
}
```

---

## Troubleshooting

###  "Model not found" Error

**Problem:** Error says model doesn't exist at path

**Solution:**
1. Train the model first:
   ```bash
   python scripts/train_with_real_data.py --mode multi --timesteps 50000 --no-reward-normalization
   ```

2. Or manually verify path:
   ```bash
   # Windows
   dir models\real_data\multi_company\PPO_seed44\final_model.zip

   # Mac/Linux
   ls models/real_data/multi_company/PPO_seed44/final_model.zip
   ```

###  "Port 5000 already in use"

**Problem:** Another application is using port 5000

**Solution:**
Option A: Kill the process using port 5000
```bash
# Windows
netstat -ano | findstr :5000

# Mac/Linux
lsof -i :5000
```

Option B: Use a different port
```bash
# Edit optimizer_app.py, change:
# app.run(debug=True, port=5000)
# to:
# app.run(debug=True, port=5001)
```

###  "ImportError: No module named 'flask'"

**Problem:** Required packages not installed

**Solution:**
```bash
pip install -r requirements.txt
```

###  Results don't look right

**Problem:** Negative rewards or unexpected actions

**Checklist:**
1.  Is the model trained? (Should converge to 0.87+ reward)
2.  Are cash flows positive? (Model expects operating cash flow)
3.  Is debt reasonable? (Not >500% of equity)
4.  Is the horizon reasonable? (1-10 quarters)

---

## Advanced Usage

### Batch Testing

Save to `test_batch.json`:
```json
{
  "companies": [
    {
      "ticker": "TECH1",
      "debt": 50,
      "equity": 300,
      "cash": 100,
      "cf_history": [0.10, 0.12, 0.15, 0.18, 0.20]
    },
    {
      "ticker": "TECH2",
      "debt": 75,
      "equity": 250,
      "cash": 80,
      "cf_history": [0.08, 0.10, 0.13, 0.16, 0.19]
    }
  ],
  "steps": 5
}
```

Create `batch_optimizer.py`:
```python
import json
import requests

with open('test_batch.json') as f:
    batch = json.load(f)

for company in batch['companies']:
    response = requests.post(
        'http://localhost:5000/api/optimize',
        json={**company, 'steps': batch['steps']}
    )
    result = response.json()
    print(f"{company['ticker']}: Total Reward = {result['total_reward']:.4f}")
```

Run:
```bash
python batch_optimizer.py
```

### Integration with Corporate Systems

```python
# Connect to your database/ERP
from optimizer_client import CapitalStructureOptimizer

optimizer = CapitalStructureOptimizer('http://localhost:5000')

for ticker in get_company_list():
    data = fetch_financial_data(ticker)
    result = optimizer.optimize(
        ticker=ticker,
        debt=data['total_debt'],
        equity=data['shareholders_equity'],
        cash=data['cash_equivalents'],
        cf_history=data['last_5q_cf'],
        steps=4
    )
    save_recommendation(ticker, result)
```

---

## Performance Metrics

### Expected Model Performance

**Training Convergence:**
-  Converges in: 5,000-10,000 steps
-  Final reward: 0.87-0.88
-  Training time: 5-10 minutes

**Inference Performance:**
-  Single optimization: ~50-100ms
-  Can optimize 100 companies in: ~10 seconds
-  Memory usage: ~500MB

### Model Architecture

```
PPO (Proximal Policy Optimization)
 Policy Network (3 layers)
    256 neurons
    ReLU activation
    Output: Action (continuous, [-1, 1])
 Value Network (3 layers)
    256 neurons
    Output: State value estimate
 Training
     n_steps: 128
     batch_size: 128
     learning_rate: 1e-3
     entropy_coef: 0.05
```

### Reward Components

```
Total Reward =
  + 0.3 × Value Realization
  + 0.2 × Flexibility Bonus
  + 0.2 × Optimization Score
  - 0.1 × Distress Penalty
  - 0.05 × Transaction Costs
  + 0.05 × Survival Bonus
```

---

##  Training Alternative Algorithms: SAC & TD3

While **PPO** is the default model, you can also train **SAC** (Soft Actor-Critic) or **TD3** (Twin Delayed DDPG) on your real data.

### Why SAC?
-  **Sample efficient**: Learns faster with fewer interactions
-  **Stable**: Off-policy learning with entropy regularization
-  **Continuous control**: Naturally handles continuous actions
-  Slower wall-clock time due to more gradient updates

### Training SAC

#### Quick Test (10k steps)
```bash
python scripts/train_with_real_data.py \
  --mode single \
  --ticker AAPL \
  --algorithm SAC \
  --timesteps 10000 \
  --no-reward-normalization
```

**Output:**
```
================================================================================
Training SAC on AAPL (10,000 timesteps)
================================================================================
 Training complete!
 Final model saved to: models/real_data/AAPL/SAC_seed42/final_model
```

#### Full Training (recommended: 500k steps)
```bash
python scripts/train_with_real_data.py \
  --mode single \
  --ticker AAPL \
  --algorithm SAC \
  --timesteps 500000 \
  --seed 42 \
  --no-reward-normalization
```

#### Multi-Company Training
```bash
python scripts/train_with_real_data.py \
  --mode multi \
  --algorithm SAC \
  --timesteps 1000000 \
  --no-reward-normalization
```

### SAC Hyperparameters

The optimized SAC config is in `configs/optimized_hyperparams.yaml`:

```yaml
SAC:
  learning_rate: 1.0e-4        # Reduced for stability
  buffer_size: 100_000         # Smaller buffer (recent experiences)
  learning_starts: 1_000       # Start learning early
  batch_size: 64               # Smaller batches for more updates
  tau: 0.02                    # Faster target network updates
  train_freq: 4                # Train every 4 steps
  gradient_steps: 4            # 4 gradient updates per step
  ent_coef: 0.2                # Fixed entropy (no auto-tuning)
  gamma: 0.99                  # Discount factor
```

**Key Improvements:**
- `learning_rate` 3e-4  1e-4: Better convergence stability
- `buffer_size` 1M  100k: Focus on recent experiences
- `learning_starts` 10k  1k: Faster learning on limited data
- `batch_size` 256  64: More frequent updates
- `tau` 0.005  0.02: Faster target network synchronization
- `train_freq` & `gradient_steps` 1  4: More training iterations

### Performance Comparison

**SAC on AAPL (50k steps, seed=42):**
- Average Reward: **0.8674**
- Std Dev: **0.0018**
- Convergence: Stable within first 10k steps

**vs PPO Baseline:**
- Both achieve similar performance (~0.87 reward)
- SAC slightly more sample efficient early on
- PPO requires fewer hyperparameter tuning

### Generating Convergence Plots

```bash
# Train and generate convergence curve
python scripts/train_sac_improved.py \
  --ticker AAPL \
  --timesteps 50000 \
  --seed 42
```

This creates:
- **PNG Plot**: `visualizations/sac_convergence_comparison_AAPL_seed42.png`
- **CSV Data**: `logs/convergence/AAPL/SAC_improved_seed42/episode_rewards.csv`
- **Model**: `models/real_data/AAPL/SAC_improved_seed42/final_model.zip`

### Using SAC in the Optimizer

To switch the optimizer to use SAC instead of PPO:

1. Train SAC model:
   ```bash
   python scripts/train_with_real_data.py \
     --mode multi \
     --algorithm SAC \
     --timesteps 500000
   ```

2. Update `optimizer_app.py` to load SAC:
   ```python
   # Line ~20 in optimizer_app.py
   MODEL_PATH = 'models/real_data/multi_company/SAC_seed42/final_model.zip'
   ```

3. Restart the optimizer:
   ```bash
   python optimizer_app.py
   ```

### Training Tips for SAC

**For faster training:**
- Increase `learning_starts` to 5000 (avoid initial random exploration)
- Increase `train_freq` to 8 (more training iterations)
- Decrease `batch_size` to 32 (more frequent updates, less memory)

**For more stable training:**
- Decrease `learning_rate` to 5e-5
- Increase `buffer_size` to 200_000
- Decrease `train_freq` to 2

**For exploration:**
- Increase `ent_coef` to 0.5 (more entropy = more exploration)
- Increase `learning_starts` to 50_000 (collect more random data)

---

## FAQ

**Q: Why is my company's leverage going up?**
A: The model might be utilizing cheap debt to finance growth. This is optimal if:
- Interest coverage > 2.0x (can afford the debt)
- Cash flows are positive and growing
- Growth opportunities exist

**Q: Why are rewards negative?**
A: This can happen if:
- Debt is already too high (distressed)
- Cash flows are declining
- Company needs to deleverage to survive

**Q: Can I retrain the model?**
A: Yes! Use:
```bash
python scripts/train_with_real_data.py --mode multi --timesteps 100000 --no-reward-normalization
```

**Q: How does this compare to real capital structure decisions?**
A: The model:
-  Uses fundamental financial metrics
-  Balances growth, stability, and risk
-  Acts consistently across scenarios
-  Doesn't account for market conditions
-  Doesn't consider regulatory constraints
-  Doesn't factor in management preferences

Use this as a **decision support tool**, not a replacement for human judgment.

**Q: Can I use this for real investment decisions?**
A: Not directly. Recommended workflow:
1. Run optimizer to get AI suggestion
2. Validate with domain experts
3. Check current market conditions
4. Consider company-specific constraints
5. Make final decision with human oversight

---

## Support & Documentation

- **Code Repository**: `src/environment/capital_structure_env.py`
- **Training Script**: `scripts/train_with_real_data.py`
- **Model Config**: `configs/optimized_hyperparams.yaml`
- **Training Data**: `data/training/real_data_dataset_augmented_20.npy`

For issues:
1. Check the error message in the web interface
2. Review logs in the terminal where Flask is running
3. Verify model file exists: `models/real_data/multi_company/PPO_seed44/final_model.zip`
4. Try a simple example first (use "Quick Load" button)

---

## Summary

The Capital Structure Optimizer is a complete solution for testing capital structure optimization using deep reinforcement learning. Use it to:

 Validate that the model works correctly
 Test different company scenarios
 Understand AI recommendations
 Generate insights for capital structure decisions

**Get started now:**
```bash
python optimizer_app.py
# Then visit http://localhost:5000
```

Happy optimizing!
