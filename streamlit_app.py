

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

# Configuration Streamlit
st.set_page_config(
    page_title="Capital Structure Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Capital Structure Optimization using RL - PPO, SAC, TD3"}
)

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO, SAC, TD3
from src.utils.config import load_config
from src.environment.capital_structure_env import CapitalStructureEnv
from src.models.company import CompanyModel
from src.utils.finance import calculate_wacc, calculate_enterprise_value

# ==================== CUSTOM CSS ====================

def load_custom_css():
    """Load custom CSS for modern design"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    h2 {
        color: #a0aec0;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #cbd5e0;
        font-weight: 500;
        margin-top: 1.5rem;
    }
    
    /* Cards */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0aec0;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    /* Input Fields */
    /* Make numeric inputs display values in black for clarity */
    .stNumberInput input {
        color: #000 !important;
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        border-radius: 8px;
        padding: 0.5rem;
    }

    .stTextArea textarea {
        /* Cash flow textarea: make text black for readability */
        color: #000 !important;
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        border-radius: 8px;
        padding: 0.5rem;
    }

    .stSelectbox select {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #e0e0e0;
        padding: 0.75rem;
    }
    
    .stNumberInput input:focus,
    .stTextArea textarea:focus,
    .stSelectbox select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* DataFrames */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #e0e0e0;
        font-weight: 500;
    }
    
    /* Radio buttons */
    .stRadio > label {
        background: rgba(255, 255, 255, 0.05);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stRadio > label:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Slider */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* Custom card class */
    .custom-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
    }
    
    /* Feature card */
    .feature-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.2);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 239, 125, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== CACHE & INITIALIZATION ====================

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    config = load_config('config.yaml')
    
    model_paths = {
        'ppo': 'models/real_data/augmented/PPO_aug_no_norm_seed1/final_model',
        'sac': 'models/real_data/augmented/SAC_aug_no_norm_seed1/final_model',
        'td3': 'models/real_data/augmented/TD3_aug_no_norm_seed1/final_model'
    }
    
    for algo, path in model_paths.items():
        try:
            if os.path.exists(path + '.zip'):
                if algo == 'ppo':
                    models[algo] = PPO.load(path)
                elif algo == 'sac':
                    models[algo] = SAC.load(path)
                elif algo == 'td3':
                    models[algo] = TD3.load(path)
                st.session_state[f"{algo}_loaded"] = True
            else:
                st.session_state[f"{algo}_loaded"] = False
        except Exception as e:
            st.warning(f"❌ Erreur chargement {algo.upper()}: {str(e)}")
            st.session_state[f"{algo}_loaded"] = False
    
    return models, config

@st.cache_data
def get_example_companies():
    """Get example company profiles"""
    return {
        'tech_startup': {
            'name': '🚀 Tech Startup',
            'debt': 50.0,
            'equity': 300.0,
            'cash': 100.0,
            'cf_history': [0.05, 0.08, 0.10, 0.12, 0.15],
            'description': 'Jeune entreprise tech en croissance rapide'
        },
        'mature_company': {
            'name': '🏢 Entreprise Mature',
            'debt': 200.0,
            'equity': 400.0,
            'cash': 50.0,
            'cf_history': [0.20, 0.20, 0.20, 0.20, 0.20],
            'description': 'Entreprise stable avec flux de trésorerie constants'
        },
        'growth_company': {
            'name': '📈 Entreprise en Croissance',
            'debt': 100.0,
            'equity': 200.0,
            'cash': 80.0,
            'cf_history': [0.10, 0.15, 0.20, 0.25, 0.30],
            'description': 'Entreprise à forte croissance'
        },
        'distressed': {
            'name': '⚠️ Entreprise en Difficulté',
            'debt': 300.0,
            'equity': 100.0,
            'cash': 10.0,
            'cf_history': [0.05, 0.03, 0.02, -0.01, 0.01],
            'description': 'Entreprise en détresse financière'
        },
        'apple_like': {
            'name': '💎 Apple (AAPL) - Données Réelles',
            'debt': 120.0,
            'equity': 280.0,
            'cash': 150.0,
            'cf_history': [0.25, 0.27, 0.28, 0.30, 0.32],
            'description': 'Apple Inc. - Géant technologique',
            'ticker': 'AAPL'
        },
        'google_like': {
            'name': '🔍 Google (GOOGL) - Données Réelles',
            'debt': 8.0,
            'equity': 1200.0,
            'cash': 110.0,
            'cf_history': [0.22, 0.24, 0.26, 0.28, 0.30],
            'description': 'Alphabet/Google Inc. - Leader de la recherche',
            'ticker': 'GOOGL'
        }
    }

@st.cache_data(ttl=3600)
def fetch_real_company_data(ticker: str):
    """Fetch real company data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Récupérer les données financières
        debt = float(info.get('totalDebt', 0)) / 1e9  # Convertir en milliards
        equity = float(info.get('marketCap', 0)) / 1e9
        cash = float(info.get('totalCash', 0)) / 1e9
        
        # Récupérer l'historique de cash flows (estimé à partir du rendement)
        hist = stock.quarterly_financials
        
        if hist is not None and not hist.empty and 'Operating Cash Flow' in hist.index:
            cf_values = hist.loc['Operating Cash Flow'].head(5).values / 1e9
            cf_history = [float(x) if not np.isnan(x) else 0.05 for x in cf_values][::-1]
            # Normaliser aux 5 derniers trimestres
            if len(cf_history) < 5:
                cf_history = cf_history + [cf_history[-1] if cf_history else 0.05] * (5 - len(cf_history))
        else:
            # Cash flow par défaut si non disponible
            cf_history = [0.2, 0.22, 0.24, 0.26, 0.28]
        
        return {
            'debt': max(0, debt),
            'equity': max(1, equity),
            'cash': max(0, cash),
            'cf_history': [max(0.01, x) for x in cf_history[:5]],  # Assurer valeurs positives
        }, None
    except Exception as e:
        return None, str(e)

def get_company_profile(preset_key, use_real_data=False):
    """Get company profile with option to fetch real data"""
    examples = get_example_companies()
    preset = examples[preset_key]
    
    # Si real data activé et ticker disponible
    if use_real_data and 'ticker' in preset:
        ticker = preset['ticker']
        real_data, error = fetch_real_company_data(ticker)
        
        if real_data is not None:
            return {
                **preset,
                **real_data,
                'data_source': f'✅ Données réelles ({ticker})'
            }
        else:
            return {
                **preset,
                'data_source': f'⚠️ Données réelles indisponibles (erreur: {error}), utilisant valeurs par défaut'
            }
    
    return {
        **preset,
        'data_source': '📋 Données d\'exemple'
    }

# ==================== HELPER FUNCTIONS ====================

def run_optimization(company_data, steps, model_type, models, config):
    """Run optimization with specified model"""
    if model_type not in models:
        return None, f"Modèle {model_type} non chargé"
    
    model = models[model_type]
    
    try:
        cf_data = np.array(company_data.get('cf_history', [0.1]*5))
        
        env = CapitalStructureEnv(
            config=config,
            max_steps=steps,
            scenario='baseline',
            real_cf_data=cf_data,
            disable_reward_normalization=True
        )
        
        env.company.debt = company_data.get('debt', 100.0)
        env.company.equity = company_data.get('equity', 200.0)
        env.company.cash = company_data.get('cash', 50.0)
        env.company.cf = cf_data[0] if len(cf_data) > 0 else 0.1
        
        results = []
        previous_state = {
            'debt': company_data.get('debt', 100.0),
            'equity': company_data.get('equity', 200.0),
            'cash': company_data.get('cash', 50.0),
            'leverage': company_data.get('debt', 100.0) / (company_data.get('debt', 100.0) + company_data.get('equity', 200.0)),
            'wacc': 0.0,
        }
        
        obs, _ = env.reset()
        
        for step in range(steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            raw_reward = info.get('raw_reward', reward)
            company = env.company
            
            # Calculer les deltas (changements)
            debt_delta = float(company.debt) - previous_state['debt']
            equity_delta = float(company.equity) - previous_state['equity']
            cash_delta = float(company.cash) - previous_state['cash']
            leverage_delta = float(company.get_leverage()) - previous_state['leverage']
            
            # Interpréter l'action en décision
            action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            
            # Actions RL sont généralement dans [-1, 1]
            # On les interprète en termes de changement de structure de capital
            if action_val > 0.5:
                decision = "📈 Augmenter Dettes"
            elif action_val < -0.5:
                decision = "📉 Réduire Dettes"
            else:
                decision = "➡️ Structure Optimale - Maintenir"
            
            result = {
                'step': step + 1,
                'action_value': float(action_val),
                'decision': decision,
                'reward': float(raw_reward),
                'raw_reward': float(raw_reward),
                'debt': float(company.debt),
                'debt_delta': float(debt_delta),
                'equity': float(company.equity),
                'equity_delta': float(equity_delta),
                'cash': float(company.cash),
                'cash_delta': float(cash_delta),
                'leverage': float(company.get_leverage()),
                'leverage_delta': float(leverage_delta),
                'interest_coverage': float(company.get_interest_coverage()),
                'wacc': float(company.calculate_wacc()),
                'cf': float(company.cf),
            }
            results.append(result)
            
            # Mettre à jour l'état précédent
            previous_state = {
                'debt': float(company.debt),
                'equity': float(company.equity),
                'cash': float(company.cash),
                'leverage': float(company.get_leverage()),
                'wacc': float(company.calculate_wacc()),
            }
            
            if done or truncated:
                break
        
        return results, "Succès"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Erreur: {str(e)}"

def create_summary_metrics(results, initial_data):
    """Create summary metrics"""
    if not results:
        return {}
    
    final = results[-1]
    
    return {
        'Total Reward': sum(r['reward'] for r in results),
        'Avg Reward': sum(r['reward'] for r in results) / len(results),
        'Debt Change': final['debt'] - initial_data['debt'],
        'Equity Change': final['equity'] - initial_data['equity'],
        'Leverage Change': final['leverage'] - (initial_data['debt'] / (initial_data['debt'] + initial_data['equity'])),
        'WACC Final': final['wacc'],
        'Interest Coverage': final['interest_coverage'],
    }

def detect_convergence(results, threshold=0.05):
    """Detect if model has converged based on reward stability"""
    if len(results) < 3:
        return False, 0.0
    
    # Calculer la variance des 3 dernières récompenses
    last_rewards = [r['reward'] for r in results[-3:]]
    avg_reward = np.mean(last_rewards)
    
    if avg_reward == 0:
        variance = 0
    else:
        variance = np.std(last_rewards) / (abs(avg_reward) + 1e-6)
    
    # Convergé si la variance est faible
    is_converged = variance < threshold
    
    return is_converged, variance

# ==================== PAGE: HOME ====================

def page_home():
    """Home page"""
    # Load models first
    if 'models' not in st.session_state:
        st.session_state.models, st.session_state.config = load_models()
    
    models = st.session_state.models
    
    st.markdown("# 🚀 Capital Structure Optimization Dashboard")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Modèles Disponibles", len(models))
    
    with col2:
        st.metric("🤖 Algorithmes", "PPO, SAC, TD3")
    
    with col3:
        st.metric("📈 Version", "2.0")
    
    st.markdown("---")
    
    # Feature cards
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Vue d'ensemble
    
    Cette plateforme vous permet d'optimiser la structure de capital de votre entreprise en utilisant 
    des algorithmes d'apprentissage par renforcement de pointe.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("""
        #### ⚙️ Optimisation
        Utilisez PPO, SAC ou TD3 pour optimiser
        la structure de capital en temps réel
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("""
        #### 🔄 Comparaison
        Comparez les performances des
        trois algorithmes simultanément
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("""
        #### 📊 Analytics
        Visualisez les métriques et
        téléchargez les résultats
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 📚 Technologies & Métriques
    
    **Algorithmes RL**
    - **PPO** (Proximal Policy Optimization) - Stable et performant
    - **SAC** (Soft Actor-Critic) - Exploration optimale
    - **TD3** (Twin Delayed DDPG) - Robuste et précis
    
    **Métriques Optimisées**
    - Ratio de levier (Debt/Equity)
    - WACC (Coût moyen pondéré du capital)
    - Couverture d'intérêts
    - Valeur d'entreprise
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE: OPTIMIZATION ====================

def page_optimization():
    """Optimization page"""
    st.markdown("# ⚙️ Optimisation de Structure de Capital")
    st.markdown("---")
    
    if 'models' not in st.session_state:
        st.session_state.models, st.session_state.config = load_models()
    
    models = st.session_state.models
    config = st.session_state.config
    
    if not models:
        st.error("❌ Aucun modèle n'a pu être chargé. Vérifiez les chemins des modèles.")
        return
    
    examples = get_example_companies()
    
    with st.sidebar:
        st.markdown("## 🏢 Sélection Entreprise")
        
        company_preset = st.selectbox(
            "Sélectionner un profil prédéfini:",
            list(examples.keys()),
            format_func=lambda x: examples[x]['name']
        )
        
        st.info(f"📝 {examples[company_preset]['description']}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 💰 Données Entreprise")
        
        preset = get_company_profile(company_preset)
        
        # Afficher la source des données
        st.info(preset.get('data_source', '📋 Données d\'exemple'))
        
        # Option pour charger les vraies données (si disponible)
        if 'ticker' in get_example_companies()[company_preset]:
            use_real = st.checkbox("📡 Charger vraies données de l'API", value=False)
            if use_real:
                with st.spinner(f"📡 Récupération des données {get_example_companies()[company_preset]['ticker']}..."):
                    preset = get_company_profile(company_preset, use_real_data=True)
                st.info(preset.get('data_source', ''))
        
        debt = st.number_input("Dettes (M$)", value=preset['debt'], min_value=0.0)
        equity = st.number_input("Capitaux Propres (M$)", value=preset['equity'], min_value=0.1)
        cash = st.number_input("Trésorerie (M$)", value=preset['cash'], min_value=0.0)
        
        st.markdown("### 📊 Historique Cash Flow")
        cf_input = st.text_area(
            "Cash flows (séparés par des virgules):",
            value=", ".join(map(str, preset['cf_history']))
        )
        
        try:
            cf_history = [float(x.strip()) for x in cf_input.split(',') if x.strip()]
        except:
            cf_history = preset['cf_history']
        
        steps = st.slider("Nombre d'étapes", 1, 20, 5)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 Sélection Modèle")
        
        available_models = list(models.keys())
        selected_model = st.selectbox(
            "Choisir un modèle:",
            available_models,
            format_func=lambda x: x.upper()
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.success(f"✓ **{selected_model.upper()}** chargé")
        with col_b:
            if st.button("🔄 Rafraîchir", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("▶️ Lancer l'optimisation", use_container_width=True, type="primary"):
        with st.spinner(f"⏳ Optimisation en cours avec {selected_model.upper()}..."):
            company_data = {
                'debt': debt,
                'equity': equity,
                'cash': cash,
                'cf_history': cf_history,
                'ticker': company_preset.upper()
            }
            
            results, msg = run_optimization(company_data, steps, selected_model, models, config)
            
            if results:
                st.success(f"✅ {msg}")
                st.session_state.last_results = results
                st.session_state.last_company = company_data
            else:
                st.error(f"❌ {msg}")
    
    if 'last_results' in st.session_state and st.session_state.last_results:
        results = st.session_state.last_results
        company_data = st.session_state.last_company
        
        st.markdown("---")
        st.markdown("## 📈 Résultats & Décision Finale")
        
        metrics = create_summary_metrics(results, company_data)
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("💰 Récompense Totale", f"{metrics['Total Reward']:.4f}")
        with metric_cols[1]:
            st.metric("📊 Récompense Moyenne", f"{metrics['Avg Reward']:.4f}")
        with metric_cols[2]:
            st.metric("📉 Changement Levier", f"{metrics['Leverage Change']:.4f}")
        with metric_cols[3]:
            st.metric("🎯 WACC Final", f"{metrics['WACC Final']:.4f}")
        
        st.markdown("---")
        
        # Détection de convergence
        is_converged, variance = detect_convergence(results)
        
        # Afficher l'état de convergence
        if is_converged:
            st.success("✅ **Le modèle a CONVERGÉ** - Résultats stables et fiables")
        else:
            st.warning(f"⚠️ Le modèle n'a pas complètement convergé (variance: {variance:.4f})")
        
        st.markdown("---")
        st.markdown("### 🎯 Décision Finale Optimale")
        
        df_results = pd.DataFrame(results)
        final_result = results[-1]
        
        # Afficher SEULEMENT la décision finale
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        decision_emoji = {
            "📈 Augmenter Dettes": "📈",
            "📉 Réduire Dettes": "📉",
            "➡️ Maintenir Stabilité": "➡️"
        }
        
        final_decision = final_result['decision']
        emoji = decision_emoji.get(final_decision, "🎯")
        
        st.markdown(f"""
        ## {emoji} {final_decision}
        
        **État Final de l'Entreprise:**
        
        | Métrique | Valeur Initial | Valeur Final |
        |----------|---|---|
        | **Dettes** | ${company_data['debt']:.1f}M | ${final_result['debt']:.1f}M |
        | **Capitaux Propres** | ${company_data['equity']:.1f}M | ${final_result['equity']:.1f}M |
        | **Ratio de Levier** | {company_data['debt']/(company_data['debt']+company_data['equity']):.3f} | {final_result['leverage']:.3f} |
        
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Tableau Détaillé - Toutes les Étapes")
        
        # Créer un dataframe formaté pour affichage
        display_df = pd.DataFrame({
            'Étape': df_results['step'].astype(int),
            'Décision': df_results['decision'],
            'Dettes (M$)': df_results['debt'].apply(lambda x: f"{x:.1f}"),
            'Δ Dettes': df_results['debt_delta'].apply(lambda x: f"{x:+.2f}"),
            'Capitaux (M$)': df_results['equity'].apply(lambda x: f"{x:.1f}"),
            'Δ Capitaux': df_results['equity_delta'].apply(lambda x: f"{x:+.2f}"),
            'Levier': df_results['leverage'].apply(lambda x: f"{x:.3f}"),
            'Δ Levier': df_results['leverage_delta'].apply(lambda x: f"{x:+.4f}"),
        })
        
        st.dataframe(display_df, use_container_width=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Trajectoires", "💹 Changements", "📈 Détails Métriques", "📋 Données Brutes"])

        with tab1:
            df_results = pd.DataFrame(results)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_results['step'],
                y=df_results['reward'],
                mode='lines+markers',
                name='Récompense',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            
            fig.update_layout(
                title="Récompense par étape",
                xaxis_title="Étape",
                yaxis_title="Récompense",
                hovermode='x unified',
                template='plotly_dark',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                pass
                # fig = go.Figure()
                # 
                # # Calcul des changements cumulatifs
                # initial_debt = company_data['debt']
                # initial_equity = company_data['equity']
                # 
                # debt_change_cumul = df_results['debt'] - initial_debt
                # equity_change_cumul = df_results['equity'] - initial_equity
                # 
                # fig.add_trace(go.Scatter(
                #     x=df_results['step'],
                #     y=debt_change_cumul,
                #     mode='lines+markers',
                #     name='Δ Dettes (Cumulatif)',
                #     line=dict(color='#f56565', width=3),
                #     marker=dict(size=10),
                #     fill='tozeroy',
                #     fillcolor='rgba(245, 101, 101, 0.3)',
                #     hovertemplate='<b>Étape %{x}</b><br>Changement Dettes: %{y:+.2f}M$<extra></extra>'
                # ))
                # fig.add_trace(go.Scatter(
                #     x=df_results['step'],
                #     y=equity_change_cumul,
                #     mode='lines+markers',
                #     name='Δ Capitaux Propres (Cumulatif)',
                #     line=dict(color='#48bb78', width=3),
                #     marker=dict(size=10),
                #     fill='tozeroy',
                #     fillcolor='rgba(72, 187, 120, 0.3)',
                #     hovertemplate='<b>Étape %{x}</b><br>Changement Capitaux: %{y:+.2f}M$<extra></extra>'
                # ))
                # 
                # fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
                # 
                # fig.update_layout(
                #     title="📊 Changements Réels - Structure de Capital",
                #     xaxis_title="Étape",
                #     yaxis_title="Changement Cumulatif (M$)",
                #     hovermode='x unified',
                #     template='plotly_dark',
                #     height=350,
                #     paper_bgcolor='rgba(0,0,0,0)',
                #     plot_bgcolor='rgba(0,0,0,0)',
                #     legend=dict(x=0.01, y=0.99)
                # )
                # st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                pass
                # fig = go.Figure()
                # 
                # # Calcul du levier initial et changement cumulatif
                # initial_leverage = initial_debt / (initial_debt + initial_equity)
                # leverage_change_cumul = df_results['leverage'] - initial_leverage
                # 
                # fig.add_trace(go.Scatter(
                #     x=df_results['step'],
                #     y=leverage_change_cumul,
                #     mode='lines+markers',
                #     name='Δ Levier (Cumulatif)',
                #     line=dict(color='#ed8936', width=3),
                #     marker=dict(size=10),
                #     fill='tozeroy',
                #     fillcolor='rgba(237, 137, 54, 0.3)',
                #     hovertemplate='<b>Étape %{x}</b><br>Changement Levier: %{y:+.4f}<extra></extra>'
                # ))
                # 
                # fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
                # 
                # fig.update_layout(
                #     title="📈 Changement Réel - Ratio de Levier",
                #     xaxis_title="Étape",
                #     yaxis_title="Δ Levier (Cumulatif)",
                #     hovermode='x unified',
                #     template='plotly_dark',
                #     height=350,
                #     paper_bgcolor='rgba(0,0,0,0)',
                #     plot_bgcolor='rgba(0,0,0,0)',
                #     showlegend=True
                # )
                # st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("#### 📊 Changements Incrémentaux")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                colors = ['#48bb78' if x >= 0 else '#f56565' for x in df_results['debt_delta']]
                fig.add_trace(go.Bar(
                    x=df_results['step'],
                    y=df_results['debt_delta'],
                    name='Δ Dettes',
                    marker=dict(color=colors),
                    text=[f"{x:+.2f}" for x in df_results['debt_delta']],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Changement Dettes par Étape",
                    xaxis_title="Étape",
                    yaxis_title="Δ Dettes (M$)",
                    template='plotly_dark',
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                colors = ['#48bb78' if x >= 0 else '#f56565' for x in df_results['equity_delta']]
                fig.add_trace(go.Bar(
                    x=df_results['step'],
                    y=df_results['equity_delta'],
                    name='Δ Capitaux',
                    marker=dict(color=colors),
                    text=[f"{x:+.2f}" for x in df_results['equity_delta']],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Changement Capitaux par Étape",
                    xaxis_title="Étape",
                    yaxis_title="Δ Capitaux (M$)",
                    template='plotly_dark',
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns(2)
            
            with col3:
                fig = go.Figure()
                colors = ['#48bb78' if x >= 0 else '#f56565' for x in df_results['cash_delta']]
                fig.add_trace(go.Bar(
                    x=df_results['step'],
                    y=df_results['cash_delta'],
                    name='Δ Trésorerie',
                    marker=dict(color=colors),
                    text=[f"{x:+.2f}" for x in df_results['cash_delta']],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Changement Trésorerie par Étape",
                    xaxis_title="Étape",
                    yaxis_title="Δ Trésorerie (M$)",
                    template='plotly_dark',
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                fig = go.Figure()
                colors = ['#48bb78' if x >= 0 else '#f56565' for x in df_results['leverage_delta']]
                fig.add_trace(go.Bar(
                    x=df_results['step'],
                    y=df_results['leverage_delta'],
                    name='Δ Levier',
                    marker=dict(color=colors),
                    text=[f"{x:+.4f}" for x in df_results['leverage_delta']],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Changement Levier par Étape",
                    xaxis_title="Étape",
                    yaxis_title="Δ Levier",
                    template='plotly_dark',
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_results['step'],
                    y=df_results['wacc'],
                    mode='lines+markers',
                    name='WACC',
                    line=dict(color='#9f7aea', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(159, 122, 234, 0.2)'
                ))
                
                fig.update_layout(
                    title="Coût Moyen Pondéré du Capital (WACC)",
                    xaxis_title="Étape",
                    yaxis_title="WACC",
                    hovermode='x unified',
                    template='plotly_dark',
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_results['step'],
                    y=df_results['interest_coverage'],
                    mode='lines+markers',
                    name='Interest Coverage',
                    line=dict(color='#4299e1', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(66, 153, 225, 0.2)'
                ))
                
                fig.update_layout(
                    title="Capacité de Couverture d'Intérêts",
                    xaxis_title="Étape",
                    yaxis_title="Interest Coverage Ratio",
                    hovermode='x unified',
                    template='plotly_dark',
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.dataframe(df_results, use_container_width=True)

# ==================== PAGE: COMPARISON ====================

def page_comparison():
    """Model comparison page"""
    st.markdown("# 🔄 Comparaison des Modèles")
    st.markdown("---")
    
    if 'models' not in st.session_state:
        st.session_state.models, st.session_state.config = load_models()
    
    models = st.session_state.models
    config = st.session_state.config
    
    if len(models) < 2:
        st.error("❌ Au moins 2 modèles doivent être chargés pour une comparaison.")
        return
    
    examples = get_example_companies()
    
    with st.sidebar:
        st.markdown("## 🏢 Configuration Comparaison")
        company_preset = st.selectbox(
            "Profil entreprise:",
            list(examples.keys()),
            format_func=lambda x: examples[x]['name'],
            key="comp_company"
        )
    
    preset = examples[company_preset]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 💰 Données Entreprise")
        
        # Option pour charger les vraies données (si disponible)
        if 'ticker' in examples[company_preset]:
            use_real = st.checkbox("📡 Charger vraies données de l'API", value=False, key="comp_real_data")
            if use_real:
                with st.spinner(f"📡 Récupération des données {examples[company_preset]['ticker']}..."):
                    preset = get_company_profile(company_preset, use_real_data=True)
                st.info(preset.get('data_source', ''))
        
        debt = st.number_input("Dettes (M$)", value=preset['debt'], key="comp_debt")
        equity = st.number_input("Capitaux Propres (M$)", value=preset['equity'], key="comp_equity", min_value=0.1)
        cash = st.number_input("Trésorerie (M$)", value=preset['cash'], key="comp_cash")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Paramètres")
        cf_input = st.text_area(
            "Cash flows:",
            value=", ".join(map(str, preset['cf_history'])),
            key="comp_cf"
        )
        steps = st.slider("Étapes", 1, 20, 5, key="comp_steps")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("▶️ Comparer les modèles", use_container_width=True, type="primary"):
        try:
            cf_history = [float(x.strip()) for x in cf_input.split(',') if x.strip()]
        except:
            cf_history = preset['cf_history']
        
        company_data = {
            'debt': debt,
            'equity': equity,
            'cash': cash,
            'cf_history': cf_history,
            'ticker': company_preset.upper()
        }
        
        st.markdown("---")
        st.markdown("## 📊 Résultats de Comparaison")
        
        comparison_results = {}
        
        with st.spinner("⏳ Exécution comparaison..."):
            for algo in models.keys():
                with st.spinner(f"  → {algo.upper()}..."):
                    results, msg = run_optimization(company_data, steps, algo, models, config)
                    if results:
                        comparison_results[algo] = results
        
        if comparison_results:
            st.success("✅ Comparaison complétée!")
            
            st.markdown("### 📈 Résumé")
            
            summary_data = []
            for algo, results in comparison_results.items():
                metrics = create_summary_metrics(results, company_data)
                summary_data.append({
                    'Modèle': algo.upper(),
                    'Récompense Totale': f"{metrics['Total Reward']:.4f}",
                    'Récompense Moyenne': f"{metrics['Avg Reward']:.4f}",
                    'Levier Final': f"{results[-1]['leverage']:.4f}",
                    'WACC Final': f"{metrics['WACC Final']:.4f}"
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            tab1, tab2, tab3 = st.tabs(["🏆 Récompenses", "📊 Levier", "💹 WACC"])
            
            colors = {'ppo': '#667eea', 'sac': '#f56565', 'td3': '#48bb78'}
            
            with tab1:
                fig = go.Figure()
                for algo, results in comparison_results.items():
                    df = pd.DataFrame(results)
                    fig.add_trace(go.Scatter(
                        x=df['step'],
                        y=df['reward'],
                        mode='lines+markers',
                        name=algo.upper(),
                        line=dict(color=colors.get(algo, '#667eea'), width=2)
                    ))
                
                fig.update_layout(
                    title="Comparaison des Récompenses",
                    xaxis_title="Étape",
                    yaxis_title="Récompense",
                    hovermode='x unified',
                    template='plotly_dark',
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = go.Figure()
                for algo, results in comparison_results.items():
                    df = pd.DataFrame(results)
                    fig.add_trace(go.Scatter(
                        x=df['step'],
                        y=df['leverage'],
                        mode='lines+markers',
                        name=algo.upper(),
                        line=dict(color=colors.get(algo, '#667eea'), width=2)
                    ))
                
                fig.update_layout(
                    title="Comparaison des Leviers",
                    xaxis_title="Étape",
                    yaxis_title="Levier",
                    hovermode='x unified',
                    template='plotly_dark',
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig = go.Figure()
                for algo, results in comparison_results.items():
                    df = pd.DataFrame(results)
                    fig.add_trace(go.Scatter(
                        x=df['step'],
                        y=df['wacc'],
                        mode='lines+markers',
                        name=algo.upper(),
                        line=dict(color=colors.get(algo, '#667eea'), width=2)
                    ))
                
                fig.update_layout(
                    title="Comparaison des WACC",
                    xaxis_title="Étape",
                    yaxis_title="WACC",
                    hovermode='x unified',
                    template='plotly_dark',
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("❌ Erreur lors de la comparaison.")

# ==================== PAGE: DASHBOARD ====================

def page_dashboard():
    """Analytics dashboard"""
    st.markdown("# 📊 Dashboard & Analytics")
    st.markdown("---")
    
    if 'models' not in st.session_state:
        st.session_state.models, st.session_state.config = load_models()
    
    models = st.session_state.models
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤖 Modèles Actifs", len(models))
    
    with col2:
        st.metric("📚 Données", "Augmented")
    
    with col3:
        st.metric("⚙️ Normalisation", "Désactivée")
    
    with col4:
        st.metric("🎯 Status", "✅ Opérationnel")
    
    st.markdown("---")
    
    st.markdown("## 🔍 Information Modèles")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### PPO")
        st.markdown("""
        **Proximal Policy Optimization**
        
        - Type: On-Policy
        - Avantage: Stabilité et performance
        - Usage: Production
        """)
        st.success("✅ Chargé" if 'ppo' in models else "❌ Non disponible")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### SAC")
        st.markdown("""
        **Soft Actor-Critic**
        
        - Type: Off-Policy
        - Avantage: Exploration optimale
        - Usage: Cas complexes
        """)
        st.success("✅ Chargé" if 'sac' in models else "❌ Non disponible")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### TD3")
        st.markdown("""
        **Twin Delayed DDPG**
        
        - Type: Off-Policy
        - Avantage: Robustesse
        - Usage: Actions continues
        """)
        st.success("✅ Chargé" if 'td3' in models else "❌ Non disponible")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("""
    ## 📖 À propos du Projet
    
    ### 🎯 Objectif
    Optimiser la structure de capital des entreprises en utilisant des algorithmes
    d'apprentissage par renforcement (RL) de pointe.
    
    ### 🏗️ Architecture
    - **Environnement**: Capital Structure Environment (OpenAI Gym)
    - **Modèles RL**: PPO, SAC, TD3 (Stable-baselines3)
    - **Dataset**: Real data augmented (20x)
    - **Récompense**: Sans normalisation
    
    ### 📊 Métriques Optimisées
    - Levier financier (Debt/Equity)
    - WACC (Coût moyen pondéré du capital)
    - Couverture d'intérêts
    - Valeur d'entreprise
    - Cash flows disponibles
    
    ### 🚀 Performance
    - 100,000+ étapes d'entraînement
    - Convergence validée
    - Tests sur données réelles
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 💾 Export de Données")
    
    if 'last_results' in st.session_state:
        df_export = pd.DataFrame(st.session_state.last_results)
        csv = df_export.to_csv(index=False)
        
        st.download_button(
            label="📥 Télécharger derniers résultats (CSV)",
            data=csv,
            file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("ℹ️ Aucun résultat disponible. Lancez une optimisation d'abord.")

# ==================== MAIN APP ====================

def main():
    """Main app"""
    load_custom_css()
    
    with st.sidebar:
        st.markdown("# 🎯 Navigation")
        page = st.radio(
            "Sélectionnez une page:",
            ["🏠 Accueil", "⚙️ Optimisation", "🔄 Comparaison", "📊 Dashboard"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        with st.expander("ℹ️ Informations Système"):
            st.markdown("""
            **Version**: 2.0 Styled
            **Framework**: Streamlit
            **RL Library**: Stable-baselines3
            **Design**: Modern Dark Theme
            """)
        
        with st.expander("🔗 Liens Utiles"):
            st.markdown("""
            - [Documentation](./README.md)
            - [Config](./config.yaml)
            - [Modèles](./models/)
            """)
    
    if page == "🏠 Accueil":
        page_home()
    elif page == "⚙️ Optimisation":
        page_optimization()
    elif page == "🔄 Comparaison":
        page_comparison()
    elif page == "📊 Dashboard":
        page_dashboard()

if __name__ == "__main__":
    main()