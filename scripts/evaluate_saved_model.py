import os
import numpy as np
from stable_baselines3 import SAC

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/real_data/aapl/SAC_seed42/final_model.zip')
TICKER = os.environ.get('TICKER', 'AAPL')
EVAL_EPISODES = int(os.environ.get('EVAL_EPISODES', '200'))

# Load original dataset
data = np.load('data/training/real_data_dataset.npy', allow_pickle=True).item()
company = data[TICKER]
company_cf = company['cf_normalized']

# Create env
config = load_config()
env = CapitalStructureEnv(config=config, max_steps=min(252, len(company_cf)), scenario='baseline', real_cf_data=company_cf)

# Load model
print(f'Loading model: {MODEL_PATH}')
model = SAC.load(MODEL_PATH)

# Evaluate
returns = []
for ep in range(EVAL_EPISODES):
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        # Support both Gym v0.26 (obs, reward, terminated, truncated, info) and older API
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out
        total_reward += float(reward)
    returns.append(total_reward)

import numpy as np
print(f'Evaluated {EVAL_EPISODES} episodes')
print(f'Mean reward: {np.mean(returns):.4f} | Std: {np.std(returns):.4f}')

# Save returns to csv
out_dir = os.path.join('logs', 'evaluation', TICKER)
os.makedirs(out_dir, exist_ok=True)
import csv
with open(os.path.join(out_dir, 'evaluation_returns.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'return'])
    for i,r in enumerate(returns):
        writer.writerow([i, r])
print(f'Returns saved to {os.path.join(out_dir, "evaluation_returns.csv")}')
