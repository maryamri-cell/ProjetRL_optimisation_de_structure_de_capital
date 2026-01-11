#  Testing Your Trained Model

This guide shows you how to test the Capital Structure Optimizer model to verify it's actually optimizing your capital structure correctly.

---

##  Quick Start (30 seconds)

### Option 1: Web Interface
```bash
python optimizer_app.py
```
Then open: **http://localhost:5000**

### Option 2: Command-Line Test
```bash
python test_optimizer.py
```

---

##  What You'll Test

The optimizer shows whether your trained PPO model can:

 **Optimize Debt Levels** - Choose appropriate leverage for each company type
 **Handle Growth** - Increase debt when it makes financial sense
 **Manage Distress** - Reduce debt when company is at risk
 **Maximize Value** - Balance multiple financial objectives

---

##  Web Interface Features

### Input Form
- **Ticker**: Company symbol
- **Debt/Equity/Cash**: Financial position
- **Cash Flows**: 5 quarters of historical data
- **Steps**: Optimization horizon (1-10 quarters)

### Quick-Load Examples
1. **Tech Startup** - Growing company (debt: 50M, equity: 300M)
2. **Mature Company** - Stable business (debt: 200M, equity: 400M)
3. **Distressed** - In recovery (debt: 300M, equity: 100M)
4. **Growth** - High growth (debt: 100M, equity: 250M)

### Results Dashboard
- **Total Reward**: AI performance score
- **Leverage Change**: Debt adjustment
- **Quarterly Timeline**: Quarter-by-quarter decisions
- **Detailed Metrics**: Full financial data

---

##  Key Results to Look For

###  Good Signs

**For Tech Startup:**
- Rewards positive and increasing
- Leverage increasing slightly (using cheap debt for growth)
- Interest coverage > 2.0x (staying safe)
- WACC decreasing (cheaper financing)

**For Mature Company:**
- Stable rewards around 0.2-0.3
- Leverage unchanged (already optimal)
- Interest coverage stable > 3.0x
- WACC stable around 8-9%

**For Distressed:**
- Rewards negative then improving
- Leverage decreasing (reducing debt)
- Interest coverage improving toward 2.0x
- WACC trending down

###  Warning Signs

**If you see:**
-  All negative rewards  Model not converged, need more training
-  Erratic leverage jumps  Training instability
-  Coverage < 1.0x  Model putting company in danger
-  Same action every step  Model not learning

---

##  Three Test Scenarios

### Scenario 1: Can the Model Fund Growth?

**Why test this:** Growth companies need more debt to fund expansion

**Setup:**
1. Click "Growth Company" or enter:
   - Debt: $100M
   - Equity: $250M
   - CF: `0.10, 0.15, 0.20, 0.25, 0.30`
   - Steps: 5

2. Run optimization

3. Check results:
   -  **Expected**: Leverage increases slightly (funding growth)
   -  **Expected**: Rewards positive (good decisions)
   -  **Expected**: Coverage stays > 2.0x (safe debt level)

---

### Scenario 2: Can the Model Handle Distress?

**Why test this:** Distressed companies must reduce debt to survive

**Setup:**
1. Click "Distressed" or enter:
   - Debt: $300M
   - Equity: $100M
   - CF: `0.05, 0.03, 0.02, -0.01, 0.01`
   - Steps: 5

2. Run optimization

3. Check results:
   -  **Expected**: Leverage decreases significantly (deleveraging)
   -  **Expected**: Coverage improves (moving toward safety)
   -  **Acceptable**: Initial negative rewards (hard decisions)
   -  **Expected**: Rewards improve as situation stabilizes

---

### Scenario 3: Is the Model Stable & Consistent?

**Why test this:** Real decisions must be reproducible

**Setup:**
1. Enter any company data

2. Run with 5 steps, note results

3. Run same company again with 5 steps

4. Check results:
   -  **Expected**: First 5 steps identical both runs
   -  **Expected**: No erratic jumps in leverage
   -  **Expected**: Consistent decision pattern

---

##  Understanding the Metrics

### Leverage (D/(D+E))
- **What**: Debt as % of total capital
- **Range**: 0% (all equity) to 100% (all debt)
- **Good**: 20-60% depending on industry
- **Model Goal**: Adjust leverage for company's situation

### Interest Coverage (EBIT / Interest)
- **What**: Times interest expense is covered by earnings
- **Good**: > 2.0x (comfortable debt level)
- **Bad**: < 1.0x (can't pay interest, bankruptcy risk)
- **Model Goal**: Keep coverage > 2.0x

### WACC (Weighted Average Cost of Capital)
- **What**: Average cost of capital (debt + equity)
- **Lower is better**: More value created
- **Typical**: 7-10% depending on risk
- **Model Goal**: Minimize WACC

### Reward
- **What**: AI performance score for this decision
- **Range**: -1.0 to +1.0
- **Meaning**: How good was this capital structure choice?
- **Target**: 0.8-0.9 for optimal decisions

---

##  Verification Checklist

Use this checklist to verify your model is working correctly:

###  Model Loading
- [ ] Web interface shows " Model Loaded"
- [ ] Flask server starts without errors
- [ ] No "model not found" errors

###  Basic Functionality
- [ ] Can enter company data
- [ ] Can click "Optimize" button
- [ ] Results appear in real-time
- [ ] No crashes or timeouts

###  Decision Quality
- [ ] Tech Startup: leverage increases (positive test)
- [ ] Distressed: leverage decreases (positive test)
- [ ] Mature: leverage stable (negative test)
- [ ] Rewards make sense (0.2-0.9 range)

###  Financial Sanity
- [ ] Coverage never drops below 0.5x
- [ ] Leverage stays within 0-100%
- [ ] Debt & equity decrease rarely
- [ ] WACC changes make sense

###  Consistency
- [ ] Same inputs  Same outputs
- [ ] No random behavior
- [ ] Quarterly decisions follow pattern
- [ ] No extreme jumps in metrics

---

##  Example: Tech Startup Test

### Input Data
```
Ticker: TECH
Debt: $50M
Equity: $300M
Cash: $100M
CF History: 0.05, 0.08, 0.10, 0.12, 0.15
Steps: 5
```

### Expected Results

| Q | Debt | Leverage | Coverage | Reward | WACC % |
|---|------|----------|----------|--------|--------|
| 0 | 50.0 | 14.3% | 6.5x | - | 8.00 |
| 1 | 52.1 | 14.8% | 6.2x | +0.24 | 7.95 |
| 2 | 55.3 | 15.5% | 5.8x | +0.22 | 7.98 |
| 3 | 59.1 | 16.4% | 5.3x | +0.20 | 8.05 |
| 4 | 63.8 | 17.5% | 4.8x | +0.18 | 8.15 |
| 5 | 69.2 | 18.8% | 4.2x | +0.16 | 8.28 |

** Interpretation:**
- Leverage increases 14.3%  18.8% (using cheap debt for growth)
- Rewards consistently positive (good decisions)
- Coverage stays > 4.0x (very safe)
- WACC increases slightly but growth outweighs cost

---

##  Success Criteria

Your model is working correctly if:

1.  **Loads without errors**
   - Web interface accessible at localhost:5000
   - Model status shows " LOADED"

2.  **Makes reasonable decisions**
   - Growth company  increases debt
   - Distressed company  decreases debt
   - Mature company  maintains leverage

3.  **Produces consistent results**
   - Same inputs always produce same outputs
   - No erratic behavior or crashes

4.  **Maintains financial safety**
   - Interest coverage > 1.0x always
   - No negative equity (company doesn't go under)
   - Leverage stays 0-100%

5.  **Shows learning**
   - Rewards in 0.1-0.9 range
   - Better decisions over horizon
   - Adjustments make financial sense

---

##  Next Steps After Verification

### If Model Works
1. **Use in practice**: Deploy for real capital structure decisions
2. **Integrate with systems**: Connect to financial data sources
3. **Batch test**: Run on multiple companies
4. **Monitor performance**: Track how well recommendations work

### If Model Needs Improvement
1. **Check training**: Verify model actually trained correctly
   ```bash
   python scripts/train_with_real_data.py --mode multi --timesteps 100000 --no-reward-normalization
   ```

2. **Validate data**: Ensure cash flow data is realistic

3. **Adjust hyperparams**: Modify `configs/optimized_hyperparams.yaml`

4. **Add more training**: Increase timesteps to 100k-200k

5. **Review reward design**: Check if reward function is appropriate

---

##  Troubleshooting

### Web Interface won't load
```bash
# 1. Check Python is installed
python --version

# 2. Install dependencies
pip install flask flask-cors stable-baselines3

# 3. Check model exists
ls models/real_data/multi_company/PPO_seed44/final_model.zip

# 4. Try running directly
python optimizer_app.py
```

### Model shows "not loaded"
```bash
# Train the model
python scripts/train_with_real_data.py \
  --mode multi \
  --timesteps 50000 \
  --no-reward-normalization
```

### Port 5000 already in use
```bash
# Kill the process using port 5000
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux:
lsof -i :5000
kill -9 <PID>
```

### Model gives bad results
1. Check: Is training data realistic?
2. Check: Are cash flows positive?
3. Retrain: Increase timesteps to 100k+
4. Validate: Run test_optimizer.py for detailed output

---

##  Model Performance Baseline

For reference, a well-trained model should show:

- **Convergence Time**: 5,000-10,000 steps
- **Final Reward**: 0.87-0.88
- **Training Time**: 5-10 minutes
- **Inference Speed**: 50-100ms per optimization

If your results are significantly different, check:
1. Training timesteps (should be 50,000)
2. Reward normalization is disabled
3. Hardware is not throttled
4. No background tasks interfering

---

##  Conclusion

The Capital Structure Optimizer provides a complete solution for:

1. **Testing** whether your model actually optimizes capital structure
2. **Validating** decision quality across different company types
3. **Understanding** what the AI learned and why
4. **Deploying** for real-world use

**Get started in 30 seconds:**
```bash
python optimizer_app.py
# Open http://localhost:5000
# Click "Tech Startup"
# Click "Optimize"
# See the results!
```

Happy testing!
