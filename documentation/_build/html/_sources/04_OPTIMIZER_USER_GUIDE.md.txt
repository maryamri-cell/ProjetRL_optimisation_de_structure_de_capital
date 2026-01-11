#  Capital Structure Optimizer - User Guide

## Overview
The Capital Structure Optimizer is an interactive web application that uses a trained PPO (Proximal Policy Optimization) model to optimize your company's capital structure in real-time.

## Quick Start

### 1. Launch the Application

**Windows:**
```bash
run_optimizer.bat
```

**Mac/Linux:**
```bash
chmod +x run_optimizer.sh
./run_optimizer.sh
```

**Manual (Any OS):**
```bash
python optimizer_app.py
```

### 2. Access the Web Interface
Once the server starts, open your browser and navigate to:
```
http://localhost:5000
```

## Features

###  Input Section
- **Company Name/Ticker**: Enter your company identifier (e.g., AAPL, MSFT)
- **Debt ($M)**: Current debt level in millions
- **Equity ($M)**: Current equity value in millions
- **Cash Reserves ($M)**: Available cash reserves
- **Cash Flow History**: Past 5 quarters of cash flows
- **Optimization Horizon**: Number of quarters to optimize (1-20)

###  Quick Load Examples
The interface provides pre-loaded example companies:
- **Tech Startup**: High growth, low debt
- **Mature Company**: Stable cash flows, medium debt
- **Growth Company**: Increasing cash flows, moderate debt
- **Distressed**: High debt, low cash flows

###  Results Display

#### Summary Section
Shows aggregate metrics:
- **Total Reward**: Sum of rewards across all quarters
- **Average Reward/Step**: Mean reward per quarter
- **Leverage Change**: Change in debt-to-equity ratio

#### Detailed Timeline
- **Quarter**: Quarter number (Q1, Q2, etc.)
- **Reward**: PPO model's reward signal
- **Leverage**: Debt/(Debt+Equity) ratio
- **WACC**: Weighted Average Cost of Capital
- **Interest Coverage**: EBIT/Interest ratio

#### Full Results Table
Complete breakdown of all metrics by quarter:
- Debt and Equity levels
- Cash reserves
- All financial metrics
- Model decisions and their impacts

## Understanding the Results

###  What Does "Optimization" Mean?

The model optimizes for:
1. **Value Creation**: Maximizing enterprise value through optimal leverage
2. **Financial Stability**: Maintaining healthy liquidity and coverage ratios
3. **Cost of Capital**: Minimizing WACC through optimal debt-equity mix
4. **Risk Management**: Avoiding financial distress

###  Interpreting Rewards

**High Rewards (0.7+)** indicate:
- Good capital structure decisions
- Balanced leverage and flexibility
- Sustainable growth trajectory

**Low/Negative Rewards** suggest:
- Over-leveraged position
- Insufficient cash reserves
- High financial distress risk

###  Key Metrics Explained

| Metric | Meaning | Good Range |
|--------|---------|-----------|
| **Leverage** | Debt/(Debt+Equity) | 0.30-0.50 |
| **WACC (%)** | Cost of all capital | Lower is better |
| **Interest Coverage** | EBIT/Interest | > 2.5x |
| **Debt** | Total borrowings | Sustainable level |
| **Equity** | Shareholder value | Growing trajectory |

## Testing Different Scenarios

### Scenario 1: Tech Startup
1. Click "Tech Startup" example
2. Set optimization horizon to 8 quarters
3. Click "Optimize Capital Structure"
4. **Expected**: Model should gradually increase debt as growth stabilizes

### Scenario 2: Distressed Company
1. Click "Distressed" example
2. Set horizon to 5 quarters
3. Observe the model's recovery strategy
4. **Expected**: Model should reduce debt, preserve cash

### Scenario 3: Custom Scenario
1. Manually enter your company data
2. Adjust cash flow projections
3. Run optimization
4. **Expected**: Actions tailored to your specific situation

## API Endpoints

### Check Model Status
```
GET /api/model-status
```
Response: `{loaded: boolean, model_path: string}`

### Run Optimization
```
POST /api/optimize
Body: {
  ticker: string,
  debt: number,
  equity: number,
  cash: number,
  cf_history: [number, ...],
  steps: number
}
```

### Get Example Companies
```
GET /api/example-companies
```
Response: Dictionary of example companies with data

### Compare Policies
```
POST /api/compare-policies
Body: {
  debt: number,
  equity: number,
  cash: number,
  cf_history: [number, ...]
}
```

## Troubleshooting

###  "Model Not Loaded"
**Solution**: Ensure the model exists at:
```
models/real_data/multi_company/PPO_seed44/final_model.zip
```

###  "Port 5000 Already in Use"
**Solution**: Kill the existing process or change port in `optimizer_app.py`
```python
app.run(debug=True, port=5001)  # Use different port
```

###  "Module Not Found"
**Solution**: Ensure Python path includes project root:
```bash
cd /path/to/ProjetRL
python optimizer_app.py
```

###  Slow Performance
**Solution**: The model is CPU-intensive. For faster results:
- Reduce optimization horizon (5-7 quarters)
- Run on machine with good CPU
- Consider GPU acceleration for larger batches

## Advanced Usage

### Batch Testing (Python Script)
```python
from optimizer_app import state

# Load model
results, msg = state.optimize_capital_structure(
    company_data={
        'debt': 100,
        'equity': 200,
        'cash': 50,
        'cf_history': [0.1, 0.1, 0.1, 0.1, 0.1]
    },
    steps=5
)

# Analyze results
for result in results:
    print(f"Q{result['step']}: Leverage={result['leverage']:.4f}, Reward={result['reward']:.4f}")
```

### Integration with Your Systems
The optimizer can be integrated into:
- Financial planning tools
- Corporate finance systems
- Risk management dashboards
- Real-time trading systems

## Model Details

**Architecture**: PPO (Proximal Policy Optimization)
- **Training Data**: 220 quarters of real S&P 500 companies
- **State Space**: Financial metrics + company parameters
- **Action Space**: Continuous debt/equity adjustments
- **Reward Signal**: Multi-component (value, stability, optimization)
- **Training Status**: Fully converged (reward=0.873)

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Convergence** | 5-7k steps |
| **Final Reward** | 0.873 |
| **Training Data** | 220 quarters |
| **Inference Time** | <10ms per step |
| **Accuracy** | Depends on input data quality |

## Support & Documentation

- **Model Training Details**: See `TRAINING_REPORT_IMPROVED.md`
- **Reward Analysis**: See `REWARD_NORMALIZATION_ANALYSIS.md`
- **Architecture**: See `IMPLEMENTATION_GUIDE.md`

---

**Last Updated**: December 25, 2025
**Version**: 1.0
**Status**: Production Ready
