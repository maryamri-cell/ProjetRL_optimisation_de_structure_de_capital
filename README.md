# Optimisation Dynamique de la Structure de Capital via Deep Reinforcement Learning

## 📋 Vue d'ensemble

Ce projet implémente une approche novatrice utilisant le **Deep Reinforcement Learning (RL)** pour optimiser dynamiquement la structure de capital d'une entreprise. Contrairement aux modèles statiques traditionnels, notre agent RL apprend à prendre des décisions adaptatives (émission de dette, rachats d'actions, dividendes) en fonction des conditions de marché changeantes.

### Caractéristiques principales

- ✅ **3 Algorithmes RL**: PPO, SAC, TD3
- ✅ **Environnement personnalisé** basé sur Gymnasium
- ✅ **Modèle économique complet** avec simulation réaliste
- ✅ **4 Politiques de benchmark** pour comparaison
- ✅ **5 Scénarios économiques** (baseline, récession, boom, crise de crédit, haute volatilité)
- ✅ **Métriques financières avancées** (WACC, couverture d'intérêts, rating de crédit)
- ✅ **Évaluation statistique** robuste

## 🏗️ Architecture du Projet

```
ProjetRL/
├── src/
│   ├── __init__.py
│   ├── environment/          # Environnement Gymnasium
│   │   ├── capital_structure_env.py
│   │   └── __init__.py
│   ├── models/               # Modèles économiques
│   │   ├── company.py
│   │   └── __init__.py
│   ├── agents/               # Agents RL et benchmarks
│   │   ├── rl_agents.py
│   │   ├── baselines.py
│   │   └── __init__.py
│   └── utils/                # Utilitaires
│       ├── config.py
│       ├── finance.py
│       └── __init__.py
├── train.py                  # Script d'entraînement
├── evaluate.py               # Script d'évaluation
├── config.yaml               # Configuration
├── requirements.txt          # Dépendances
├── notebooks/                # Jupyter notebooks
├── data/                     # Données
├── logs/                     # Logs d'entraînement
├── models/                   # Modèles sauvegardés
└── results/                  # Résultats et graphiques
```

## 🚀 Installation

### Prérequis
- Python 3.10+
- GPU NVIDIA (optionnel mais recommandé)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 📊 Configuration

Modifiez `config.yaml` pour ajuster:
- **Paramètres de l'environnement** (cash flows initiaux, dette, equity)
- **Hyperparamètres RL** (learning rate, batch size, etc.)
- **Scénarios économiques** (volatilité, taux, spreads)
- **Pondération des récompenses** (valeur, flexibilité, détresse)

## 🎯 Utilisation

### Entraînement

```bash
# Entraîner PPO
python train.py --algorithm PPO --scenario baseline --timesteps 500000

# Entraîner tous les algorithmes
python train.py --algorithm all --scenario baseline

# Entraîner en récession
python train.py --algorithm SAC --scenario recession --timesteps 500000
```

### Évaluation et Comparaison

```bash
# Comparer tous les algorithmes
python evaluate.py --scenario baseline --episodes 10

# Sauvegarder les résultats
python evaluate.py --scenario baseline --save-path results/baseline_eval

# Tester en récession
python evaluate.py --scenario recession --episodes 20
```

## 📐 Formulation Mathématique

### MDP (Markov Decision Process)

**États**: sf={CF_t, D_t, E_t, C_t, Leverage_t, InterestCoverage_t, ...}

**Actions**: a_t = (ΔD_t, ΔE_t, Div_t) ∈ [-1, 1]³

**Récompense**:
```
r_t = α·V_t + β·FlexScore_t - γ·DistressCost_t - δ·TransCost_t
```

Où:
- α=0.6: Poids de la valeur d'entreprise
- β=0.2: Poids de la flexibilité financière
- γ=0.15: Poids des coûts de détresse
- δ=0.05: Poids des coûts de transaction

### Fonctions clés

**WACC** (Coût Moyen Pondéré du Capital):
```
WACC = (E/(D+E))·r_e + (D/(D+E))·r_d·(1-T_c)
```

**Valeur d'Entreprise** (DCF simplifié):
```
V = Σ CF_t·(1+g)^t / (1+WACC)^t + Terminal Value
```

**Coûts de Détresse**:
```
DC(leverage) = 0.05·e^(3·(leverage-0.3)) si leverage > 0.3
```

## 📈 Algorithmes RL

### 1. **PPO (Proximal Policy Optimization)**
```
L^CLIP(θ) = E_t[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]
```
- **Avantages**: Stable, facile à tuner
- **Paramètres**: n_steps=2048, clip_range=0.2

### 2. **SAC (Soft Actor-Critic)**
```
J(π) = E[Σ r(s_t, a_t) + α·H(π(·|s_t))]
```
- **Avantages**: Off-policy, exploration naturelle
- **Paramètres**: tau=0.005, entropy coefficient=auto

### 3. **TD3 (Twin Delayed DDPG)**
- **Avantages**: Haute performance continue
- **Paramètres**: policy_delay=2, noise=0.1

## 🎲 Politiques de Benchmark

### 1. **Target Leverage**
Maintient un ratio d'endettement constant (D/E = 0.67)

### 2. **Pecking Order**
Hiérarchie: Cash interne → Dette → Equity

### 3. **Market Timing**
Émet quand marché favorable (faibles spreads, hauts P/B)

### 4. **Dynamic Trade-off**
Équilibre bénéfices fiscaux et coûts de détresse

## 📊 Scénarios Économiques

| Scénario | CF Growth | CF Vol | Rate Shock | Spread Shock |
|----------|-----------|--------|-----------|--------------|
| Baseline | 3% | 15% | 0% | 0% |
| Recession | -2% | 25% | +1% | +2% |
| Boom | 6% | 12% | -1% | -1% |
| Credit Crisis | 1% | 30% | +2% | +5% |
| High Vol | 3% | 30% | 0% | +1% |

## 📊 Métriques d'Évaluation

### Performance Financière
- Valeur d'entreprise
- Rendement total des actionnaires (TSR)
- Volatilité de la valeur
- Probabilité de faillite

### Efficience
- Distance du leverage optimal
- WACC moyen
- Vitesse d'ajustement
- Utilisation de la capacité de dette

### Robustesse
- Performance multi-régimes
- Sensibilité aux paramètres
- Stabilité de la politique

## 🔍 Résultats Attendus

### Hypothèses
- **H1**: Agent RL surpasse les benchmarks statiques en valeur (+5%)
- **H2**: Meilleure adaptation aux chocs économiques
- **H3**: Exploitation efficace du market timing
- **H4**: Maintien supérieur de la flexibilité financière

### Analyses Prévues
1. **Feature Importance** (SHAP values)
2. **Extraction de règles** interprétables
3. **Clustering** des états économiques
4. **Backtesting** sur entreprises S&P 500
5. **Analyse de sensibilité** robuste

## 🛠️ Personnalisation

### Ajouter un nouvel algorithme

```python
from stable_baselines3 import A2C

class A2CAgent(RLAgent):
    def __init__(self, env, config, model_save_path="models"):
        super().__init__(env, config, "A2C", model_save_path)
        self.model = A2C('MlpPolicy', env, verbose=1)
```

### Ajouter un scénario économique

```yaml
SCENARIOS:
  custom_scenario:
    cf_growth_mean: 0.04
    cf_volatility: 0.18
    rate_shock: 0.005
    spread_shock: 0.01
```

### Modifier la fonction de récompense

Éditez `_calculate_reward()` dans `capital_structure_env.py`

## 📝 Fichiers Principaux

| Fichier | Description |
|---------|-------------|
| `src/environment/capital_structure_env.py` | Environnement Gymnasium principal |
| `src/models/company.py` | Modèle économique d'entreprise |
| `src/agents/rl_agents.py` | Implémentation des 3 algorithmes RL |
| `src/agents/baselines.py` | Politiques de benchmark |
| `src/utils/finance.py` | Fonctions financières (WACC, DCFF, etc.) |
| `train.py` | Script d'entraînement |
| `evaluate.py` | Script d'évaluation et comparaison |

## 📚 Théories Financières

Le projet s'inspire de:

1. **Modigliani-Miller (1958)**
   ```
   V_L = V_U + T_c × D
   ```

2. **Trade-off Theory**
   ```
   V* = V_U + PV(Tax Shield) - PV(Financial Distress)
   ```

3. **Pecking Order Theory (Myers, 1984)**
   Hiérarchie: Cash → Debt → Equity

## 🔗 Références

- Sutton & Barto (2018): *Reinforcement Learning: An Introduction*
- Schulman et al. (2017): *Proximal Policy Optimization Algorithms*
- Haarnoja et al. (2018): *Soft Actor-Critic: Off-policy Maximum Entropy Deep RL*
- Graham & Harvey (2001): *The Theory and Practice of Corporate Finance*

## 🤝 Contribution

Pour contribuer:
1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


