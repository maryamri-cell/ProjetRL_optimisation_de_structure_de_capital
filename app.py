"""
Interface Web Interactive Dynamique - Flask + WebSocket
Affiche l'agent en action en temps réel avec mise à jour live
"""

import os
import sys
import numpy as np
from pathlib import Path
import warnings
import json
import threading
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Prevent Flask CLI from auto-loading .env files which may be binary/corrupt on some systems
os.environ.setdefault('FLASK_SKIP_DOTENV', '1')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_config, setup_logging, ensure_directories
from src.environment import make_capital_structure_env
from src.agents import create_baseline_policy
from stable_baselines3 import PPO as SB3_PPO
import os

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from pathlib import Path
import json

# ========== CONFIGURATION FLASK ==========

app = Flask(__name__)
CORS(app)

# Données globales partagées
class SimulationState:
    def __init__(self):
        self.running = False
        self.paused = False
        self.data = {
            'steps': [],
            'rewards': [],
            'cum_rewards': [],
            'leverages': [],
            'coverages': [],
            'debts': [],
            'equities': [],
            'values': [],
            'ratings': [],
            'actions': [],
            'metrics': {}
        }
        self.current_step = 0
        self.total_reward = 0
        self.lock = threading.Lock()
    
    def reset(self):
        with self.lock:
            self.data = {
                'steps': [],
                'rewards': [],
                'cum_rewards': [],
                'leverages': [],
                'coverages': [],
                'debts': [],
                'equities': [],
                'values': [],
                'ratings': [],
                'actions': [],
                'metrics': {}
            }
            self.current_step = 0
            self.total_reward = 0
    
    def add_step(self, step_data):
        with self.lock:
            self.data['steps'].append(step_data['step'])
            self.data['rewards'].append(step_data['reward'])
            self.total_reward += step_data['reward']
            self.data['cum_rewards'].append(self.total_reward)
            self.data['leverages'].append(step_data['leverage'])
            self.data['coverages'].append(step_data['coverage'])
            self.data['debts'].append(step_data['debt'])
            self.data['equities'].append(step_data['equity'])
            self.data['values'].append(step_data['value'])
            self.data['ratings'].append(step_data['rating'])
            self.data['actions'].append(step_data['action'])
            self.data['metrics'] = step_data['metrics']
            self.current_step = step_data['step']

state = SimulationState()

# ========== ROUTES FLASK ==========

@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_simulation():
    """Démarre une simulation"""
    data = request.json
    policy = data.get('policy', 'target_leverage')
    steps = data.get('steps', 100)
    
    state.reset()
    
    # check if a PPO model exists for this policy (so UI can notify user)
    model_found = False
    model_path = None
    try:
        if policy.lower().startswith('ppo'):
            # parse seed if provided: 'ppo_seed1' or 'ppo_seed0'
            seed = 0
            parts = policy.split('_')
            if len(parts) > 1 and parts[1].startswith('seed'):
                try:
                    seed = int(parts[1].replace('seed', ''))
                except Exception:
                    seed = 0

            candidates = [
                f"models/PPO_seed{seed}/final_model",
                f"models/PPO_seed{seed}/final_model.zip",
                f"models/PPO_seed{seed}/latest_model",
                f"models/PPO_seed{seed}/latest_model.zip",
                f"models/PPO_seed{seed}/best_model",
                f"models/PPO_seed{seed}/best_model.zip",
            ]

            for c in candidates:
                if os.path.exists(c):
                    model_found = True
                    model_path = c
                    break

    except Exception:
        model_found = False

    # Lancer la simulation en arrière-plan
    thread = threading.Thread(target=run_simulation, args=(policy, steps))
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'started', 'policy': policy, 'model_found': model_found, 'model_path': model_path})

@app.route('/api/pause', methods=['POST'])
def pause_simulation():
    """Met en pause la simulation"""
    state.paused = not state.paused
    return jsonify({'paused': state.paused})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Arrête la simulation"""
    state.running = False
    return jsonify({'status': 'stopped'})

@app.route('/api/data')
def get_data():
    """Retourne les données actuelles"""
    with state.lock:
        return jsonify(state.data)

@app.route('/api/metrics')
def get_metrics():
    """Retourne les métriques actuelles"""
    with state.lock:
        metrics = {
            'total_reward': state.total_reward,
            'current_step': state.current_step,
            'avg_reward': np.mean(state.data['rewards']) if state.data['rewards'] else 0,
            'final_leverage': state.data['leverages'][-1] if state.data['leverages'] else 0,
            'final_coverage': state.data['coverages'][-1] if state.data['coverages'] else 0,
            'final_rating': state.data['ratings'][-1] if state.data['ratings'] else 'N/A',
            'data_points': len(state.data['steps'])
        }
        return jsonify(metrics)


@app.route('/api/models')
def list_models():
    """Return available RL models found under models/PPO_seed*"""
    models_root = Path('models')
    models_list = []
    if models_root.exists():
        for p in sorted(models_root.glob('PPO_seed*')):
            # check existence of any recognized model file
            candidates = [
                p / 'final_model', p / 'final_model.zip',
                p / 'latest_model', p / 'latest_model.zip',
                p / 'best_model', p / 'best_model.zip'
            ]
            if any(c.exists() for c in candidates):
                # produce policy id used by the frontend, e.g. 'ppo_seed0'
                models_list.append({'id': p.name.lower(), 'label': p.name.replace('_', ' ').upper()})

    return jsonify(models_list)

# ========== SIMULATION ==========

def run_simulation(policy_name, max_steps):
    """Exécute une simulation complète"""
    state.running = True
    
    try:
        # Charger config
        config = load_config("config.yaml")
        
        # Créer environnement
        env = make_capital_structure_env(
            config=config,
            max_steps=max_steps,
            scenario="baseline"
        )
        
        # Créer politique
        policy = None
        rl_model = None

        # Support RL models via name convention: 'ppo_seed{N}' or 'ppo' (defaults to seed0)
        if policy_name.lower().startswith('ppo'):
            # parse seed if provided: 'ppo_seed1' or 'ppo_seed0'
            seed = 0
            parts = policy_name.split('_')
            if len(parts) > 1 and parts[1].startswith('seed'):
                try:
                    seed = int(parts[1].replace('seed', ''))
                except Exception:
                    seed = 0

            # Try common model paths
            candidates = [
                f"models/PPO_seed{seed}/final_model",
                f"models/PPO_seed{seed}/final_model.zip",
                f"models/PPO_seed{seed}/latest_model",
                f"models/PPO_seed{seed}/latest_model.zip",
                f"models/PPO_seed{seed}/best_model",
                f"models/PPO_seed{seed}/best_model.zip",
            ]

            model_path = None
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break

            # normalize: if we found a path without .zip but <path>.zip exists, prefer the zipped file
            if model_path and not model_path.endswith('.zip') and os.path.exists(model_path + '.zip'):
                model_path = model_path + '.zip'

            if model_path:
                print(f"Attempting to load RL model from: {model_path}")

            if model_path is None:
                # Fall back to baseline policy if model not found
                policy = create_baseline_policy('target_leverage', config)
            else:
                try:
                    rl_model = SB3_PPO.load(model_path, device='cpu')
                    # will use rl_model.predict(obs) inside loop
                except Exception as e:
                    print(f"Erreur chargement modèle PPO: {e}")
                    policy = create_baseline_policy('target_leverage', config)
        else:
            policy = create_baseline_policy(policy_name, config)
        
        # Réinitialiser
        obs, info = env.reset()
        
        for step in range(max_steps):
            if not state.running:
                break
            
            # Attendre si en pause
            while state.paused and state.running:
                time.sleep(0.1)
            
            # Agent prend une décision
            if rl_model is not None:
                # Use observation returned by env (obs may be tuple from gymnasium)
                obs_for_pred = obs
                try:
                    # Some envs return (obs, info) on reset; ensure proper obs
                    if isinstance(obs_for_pred, tuple) or isinstance(obs_for_pred, list):
                        obs_for_pred = obs_for_pred[0]
                except Exception:
                    pass

                action, _ = rl_model.predict(obs_for_pred, deterministic=True)
            else:
                action = policy.get_action(
                    cf=info.get('cash_flow', 0),
                    debt=info.get('debt', 0),
                    equity=info.get('equity', 0),
                    cash=info.get('cash', 0),
                    leverage=info.get('leverage', 0),
                    interest_coverage=info.get('coverage', 1.0)
                )
            
            # Environnement exécute
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Enregistrer les données
            step_data = {
                'step': step,
                'reward': float(reward),
                'leverage': float(info.get('leverage', 0)),
                'coverage': float(info.get('coverage', 1.0)),
                'debt': float(info.get('debt', 0)),
                'equity': float(info.get('equity', 0)),
                'value': float(info.get('value', 0)),
                'rating': str(info.get('rating', 'N/A')),
                'action': [float(a) for a in action],
                'metrics': {
                    'leverage_pct': f"{info.get('leverage', 0):.1%}",
                    'coverage_x': f"{info.get('coverage', 1.0):.2f}x",
                    'debt_m': f"{info.get('debt', 0):.0f}M",
                    'equity_m': f"{info.get('equity', 0):.0f}M",
                    'rating': info.get('rating', 'N/A')
                }
            }
            
            state.add_step(step_data)
            
            # Petit délai pour que le frontend puisse afficher
            time.sleep(0.05)
            
            if terminated or truncated:
                break
        
        env.close()
        state.running = False
        
    except Exception as e:
        print(f"Erreur: {e}")
        state.running = False

# ========== MAIN ==========

if __name__ == '__main__':
    ensure_directories()
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   INTERFACE INTERACTIVE - AGENT RL EN TEMPS REEL           ║
    ║                                                            ║
    ║   Ouvrez votre navigateur à:                              ║
    ║   http://localhost:5000                                   ║
    ║                                                            ║
    ║   Appuyez sur CTRL+C pour arrêter le serveur             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Créer le dossier templates s'il n'existe pas
    Path("templates").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    
    # Lancer le serveur
    app.run(debug=True, port=5000, use_reloader=False, threaded=True)
