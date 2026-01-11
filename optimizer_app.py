"""
Capital Structure Optimization Tester - Multi-Model Support
Test les modèles PPO et SAC entraînés pour l'optimisation de la structure de capital
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Skip .env file loading to avoid encoding issues
os.environ['FLASK_SKIP_DOTENV'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from stable_baselines3 import PPO, SAC, TD3
from src.utils.config import load_config
from src.environment.capital_structure_env import CapitalStructureEnv
from src.models.company import CompanyModel
from src.utils.finance import calculate_wacc, calculate_enterprise_value
import requests


app = Flask(__name__)
CORS(app)

# Global model state
class ModelState:
    def __init__(self):
        self.models = {}
        self.config = None
        self.model_paths = {
            'ppo': 'models/real_data/augmented/PPO_aug_no_norm_seed1/final_model',
            'sac': 'models/real_data/augmented/SAC_aug_no_norm_seed1/final_model',
            'td3': 'models/real_data/augmented/TD3_aug_no_norm_seed1/final_model'
        }
        self.load_models()
    
    def load_models(self):
        """Load all available trained models"""
        self.config = load_config('config.yaml')
        
        # Load PPO model
        try:
            if os.path.exists(self.model_paths['ppo'] + '.zip'):
                self.models['ppo'] = PPO.load(self.model_paths['ppo'])
                print(f"✓ PPO Model loaded from {self.model_paths['ppo']}")
            else:
                print(f"✗ PPO Model not found at {self.model_paths['ppo']}")
        except Exception as e:
            print(f"✗ Error loading PPO model: {e}")
        
        # Load SAC model
        try:
            if os.path.exists(self.model_paths['sac'] + '.zip'):
                self.models['sac'] = SAC.load(self.model_paths['sac'])
                print(f"✓ SAC Model loaded from {self.model_paths['sac']}")
            else:
                print(f"✗ SAC Model not found at {self.model_paths['sac']}")
        except Exception as e:
            print(f"✗ Error loading SAC model: {e}")

        # Load TD3 model
        try:
            if os.path.exists(self.model_paths.get('td3', '') + '.zip'):
                self.models['td3'] = TD3.load(self.model_paths['td3'])
                print(f"✓ TD3 Model loaded from {self.model_paths['td3']}")
            else:
                print(f"✗ TD3 Model not found at {self.model_paths.get('td3', '')}")
        except Exception as e:
            print(f"✗ Error loading TD3 model: {e}")

    def fetch_realtime_company(self, ticker: str):
        """Fetch company inputs from configured realtime API.

        Expected JSON response (example):
        {
            "ticker": "AAPL",
            "debt": 120.0,
            "equity": 300.0,
            "cash": 100.0,
            "cf_history": [0.25,0.27,0.28,0.30,0.32]
        }
        """
        api_cfg = self.config.get('REALTIME_API', {}) if self.config else {}
        url = api_cfg.get('url')

        def _mock_company(ticker_name: str):
            # Simple deterministic mock for local testing when realtime API is unavailable
            mock = {
                'ticker': ticker_name,
                'debt': 120.0,
                'equity': 280.0,
                'cash': 150.0,
                'cf_history': [0.25, 0.27, 0.28, 0.30, 0.32]
            }
            return mock

        # If URL is not configured or clearly a placeholder, return a mock instead of erroring
        if not url or ('example.com' in url) or ('localhost' not in url and url.startswith('http') and api_cfg.get('allow_mock_on_placeholder', True)):
            reason = 'Realtime API not configured or placeholder; using mock company data'
            return _mock_company(ticker), reason

        params = {'ticker': ticker}
        headers = {}
        api_key = api_cfg.get('api_key')
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        timeout = api_cfg.get('timeout', 5)

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            # Basic validation / extraction
            company = {
                'ticker': data.get('ticker', ticker),
                'debt': float(data.get('debt', data.get('liabilities', 0.0))),
                'equity': float(data.get('equity', data.get('market_cap', 0.0))),
                'cash': float(data.get('cash', 0.0)),
                'cf_history': [float(x) for x in data.get('cf_history', data.get('cf', []))]
            }
            return company, 'OK'
        except Exception as e:
            # On error, return a reasonable mock so the optimizer endpoint can proceed
            return _mock_company(ticker), f'Error fetching realtime API: {str(e)}; returned mock data'
    
    def optimize_capital_structure(self, company_data, steps=5, model_type='ppo'):
        """
        Run the trained model on company data to optimize capital structure
        
        Args:
            company_data: Dict with CF, debt, equity, cash, etc.
            steps: Number of quarters to optimize
            model_type: 'ppo' or 'sac'
        
        Returns:
            List of optimization steps with actions and metrics
        """
        if model_type not in self.models:
            return None, f"Model {model_type} not loaded"
        
        model = self.models[model_type]
        
        try:
            # Create temporary environment with company data
            cf_data = np.array(company_data.get('cf_history', [0.1, 0.1, 0.1, 0.1, 0.1]))
            
            env = CapitalStructureEnv(
                config=self.config,
                max_steps=steps,
                scenario='baseline',
                real_cf_data=cf_data,
                disable_reward_normalization=True
            )
            
            # Initialize company state in env
            env.company.debt = company_data.get('debt', 100.0)
            env.company.equity = company_data.get('equity', 200.0)
            env.company.cash = company_data.get('cash', 50.0)
            env.company.cf = cf_data[0] if len(cf_data) > 0 else 0.1
            
            results = []
            obs, _ = env.reset()
            
            for step in range(steps):
                # Get action from trained model
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

                # Prefer raw reward (un-normalized) when available
                raw_reward = info.get('raw_reward', reward)
                
                # Extract metrics
                company = env.company
                result = {
                    'step': step + 1,
                    'action': action.tolist() if isinstance(action, np.ndarray) else action,
                    'reward': float(raw_reward),
                    'raw_reward': float(raw_reward),
                    'normalized_reward': float(reward),
                    'debt': float(company.debt),
                    'equity': float(company.equity),
                    'cash': float(company.cash),
                    'leverage': float(company.get_leverage()),
                    'interest_coverage': float(company.get_interest_coverage()),
                    'wacc': float(company.calculate_wacc()),
                    'cf': float(company.cf),
                }
                results.append(result)
                
                if done or truncated:
                    break
            
            return results, "Success"
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"


# Initialize global state
state = ModelState()


# ============ API ENDPOINTS ============

@app.route('/')
def index():
    """Main page"""
    return render_template('optimizer.html')


@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Get model status"""
    models_info = {}
    for model_type, model in state.models.items():
        models_info[model_type] = {
            'loaded': True,
            'path': state.model_paths[model_type]
        }
    
    # Add unloaded models
    for model_type, path in state.model_paths.items():
        if model_type not in state.models:
            models_info[model_type] = {
                'loaded': False,
                'path': path
            }
    
    return jsonify({
        'models': models_info,
        'available_models': list(state.models.keys()),
        'message': f'{len(state.models)} model(s) loaded successfully'
    })


@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Run optimization on company data"""
    try:
        data = request.json
        
        # Extract company data
        company_data = {
            'debt': float(data.get('debt', 100.0)),
            'equity': float(data.get('equity', 200.0)),
            'cash': float(data.get('cash', 50.0)),
            'cf_history': [float(x) for x in data.get('cf_history', [0.1]*5)],
            'ticker': data.get('ticker', 'TEST')
        }
        
        steps = int(data.get('steps', 5))
        model_type = data.get('model_type', 'ppo')  # Default to PPO
        
        # Run optimization
        results, message = state.optimize_capital_structure(company_data, steps, model_type)
        
        if results is None:
            return jsonify({'success': False, 'message': message}), 400
        
        # Calculate summary metrics
        initial_metrics = {
            'debt': company_data['debt'],
            'equity': company_data['equity'],
            'leverage': company_data['debt'] / (company_data['debt'] + company_data['equity'])
        }
        
        final_metrics = results[-1] if results else {}
        
        improvement = {
            'debt_change': float(final_metrics.get('debt', 0)) - company_data['debt'],
            'equity_change': float(final_metrics.get('equity', 0)) - company_data['equity'],
            'leverage_change': float(final_metrics.get('leverage', 0)) - initial_metrics['leverage']
        }
        
        return jsonify({
            'success': True,
            'message': message,
            'model_type': model_type,
            'results': results,
            'summary': {
                'initial': initial_metrics,
                'final': final_metrics,
                'improvement': improvement,
                'total_reward': sum(r['reward'] for r in results),
                'avg_reward': sum(r['reward'] for r in results) / len(results) if results else 0
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


@app.route('/api/example-companies', methods=['GET'])
def example_companies():
    """Get example companies for testing"""
    examples = {
        'tech_startup': {
            'ticker': 'TECH',
            'name': 'Tech Startup',
            'debt': 50.0,
            'equity': 300.0,
            'cash': 100.0,
            'cf_history': [0.05, 0.08, 0.10, 0.12, 0.15],
            'description': 'Growing tech company with increasing cash flows'
        },
        'mature_company': {
            'ticker': 'MAT',
            'name': 'Mature Company',
            'debt': 200.0,
            'equity': 400.0,
            'cash': 50.0,
            'cf_history': [0.20, 0.20, 0.20, 0.20, 0.20],
            'description': 'Stable mature company with stable cash flows'
        },
        'distressed': {
            'ticker': 'DIST',
            'name': 'Distressed Company',
            'debt': 300.0,
            'equity': 100.0,
            'cash': 10.0,
            'cf_history': [0.05, 0.03, 0.02, -0.01, 0.01],
            'description': 'Company in financial distress'
        },
        'growth_company': {
            'ticker': 'GROWTH',
            'name': 'Growth Company',
            'debt': 100.0,
            'equity': 200.0,
            'cash': 80.0,
            'cf_history': [0.10, 0.15, 0.20, 0.25, 0.30],
            'description': 'High-growth company with increasing CF'
        },
        'apple_like': {
            'ticker': 'AAPL-LIKE',
            'name': 'Apple-like Company',
            'debt': 120.0,
            'equity': 280.0,
            'cash': 150.0,
            'cf_history': [0.25, 0.27, 0.28, 0.30, 0.32],
            'description': 'Large tech company similar to AAPL'
        }
    }
    return jsonify(examples)


@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    """Compare PPO and SAC models on same company data"""
    try:
        data = request.json
        company_data = {
            'debt': float(data.get('debt', 100.0)),
            'equity': float(data.get('equity', 200.0)),
            'cash': float(data.get('cash', 50.0)),
            'cf_history': [float(x) for x in data.get('cf_history', [0.1]*5)],
            'ticker': data.get('ticker', 'TEST')
        }
        
        steps = int(data.get('steps', 5))
        
        results = {}
        
        # Run PPO model
        if 'ppo' in state.models:
            ppo_results, msg = state.optimize_capital_structure(company_data, steps, 'ppo')
            if ppo_results:
                results['ppo'] = {
                    'results': ppo_results,
                    'total_reward': sum(r['reward'] for r in ppo_results),
                    'avg_reward': sum(r['reward'] for r in ppo_results) / len(ppo_results),
                    'final_leverage': ppo_results[-1]['leverage'],
                    'final_wacc': ppo_results[-1]['wacc'],
                    'final_debt': ppo_results[-1]['debt'],
                    'final_equity': ppo_results[-1]['equity']
                }
        
        # Run SAC model
        if 'sac' in state.models:
            sac_results, msg = state.optimize_capital_structure(company_data, steps, 'sac')
            if sac_results:
                results['sac'] = {
                    'results': sac_results,
                    'total_reward': sum(r['reward'] for r in sac_results),
                    'avg_reward': sum(r['reward'] for r in sac_results) / len(sac_results),
                    'final_leverage': sac_results[-1]['leverage'],
                    'final_wacc': sac_results[-1]['wacc'],
                    'final_debt': sac_results[-1]['debt'],
                    'final_equity': sac_results[-1]['equity']
                }

        # Run TD3 model
        if 'td3' in state.models:
            td3_results, msg = state.optimize_capital_structure(company_data, steps, 'td3')
            if td3_results:
                results['td3'] = {
                    'results': td3_results,
                    'total_reward': sum(r['reward'] for r in td3_results),
                    'avg_reward': sum(r['reward'] for r in td3_results) / len(td3_results),
                    'final_leverage': td3_results[-1]['leverage'],
                    'final_wacc': td3_results[-1]['wacc'],
                    'final_debt': td3_results[-1]['debt'],
                    'final_equity': td3_results[-1]['equity']
                }
        
        # Determine winner (algorithm with highest total_reward)
        winner = None
        if results:
            try:
                winner = max(results.keys(), key=lambda k: results[k].get('total_reward', float('-inf')))
            except Exception:
                winner = None
        
        return jsonify({
            'success': True,
            'comparison': results,
            'winner': winner,
            'initial': {
                'debt': company_data['debt'],
                'equity': company_data['equity'],
                'leverage': company_data['debt'] / (company_data['debt'] + company_data['equity'])
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/optimize-realtime', methods=['POST'])
def optimize_realtime():
    """Fetch real-time company inputs from configured API and run optimization."""
    try:
        data = request.json
        ticker = data.get('ticker')
        if not ticker:
            return jsonify({'success': False, 'message': 'ticker required'}), 400

        model_type = data.get('model_type')  # optional: ppo/sac/td3
        steps = int(data.get('steps', 5))

        company, msg = state.fetch_realtime_company(ticker)
        if company is None:
            return jsonify({'success': False, 'message': f'Error fetching realtime data: {msg}'}), 502

        # Run optimization for requested model or all available
        results = {}

        if model_type:
            if model_type not in state.models:
                return jsonify({'success': False, 'message': f'Model {model_type} not loaded'}), 400
            res, msg = state.optimize_capital_structure(company, steps, model_type)
            results[model_type] = res
        else:
            for m in state.models.keys():
                res, msg = state.optimize_capital_structure(company, steps, m)
                results[m] = res

        return jsonify({'success': True, 'ticker': ticker, 'company': company, 'results': results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    print("="*80)
    print("Capital Structure Optimization Tester - Multi-Model Flask Server")
    print("="*80)
    print(f"\nModels Loaded:")
    for model_type in ['ppo', 'sac', 'td3']:
        status = '✓ LOADED' if model_type in state.models else '✗ NOT LOADED'
        path = state.model_paths.get(model_type, 'N/A')
        print(f"  {model_type.upper()}: {status} - {path}")
    
    print(f"\nServer running at: http://localhost:5000")
    print("\nEndpoints:")
    print("  - GET  /                       Main dashboard")
    print("  - GET  /api/model-status       Check models status")
    print("  - POST /api/optimize           Run optimization (specify model_type)")
    print("  - GET  /api/example-companies  Get example companies")
    print("  - POST /api/compare-models     Compare available models (PPO, SAC, TD3)")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, port=5000, use_reloader=False)