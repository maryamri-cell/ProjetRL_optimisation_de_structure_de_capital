# Résumé du Projet

![Optimizing](images/optimizing.jpeg)

## Qu'est-ce que c'est ?

Optimisation de la structure de capital (dettes vs capitaux propres) utilisant des algorithmes d'apprentissage par renforcement (RL).

## Algorithmes Disponibles

- **PPO** (Proximal Policy Optimization)
- **SAC** (Soft Actor-Critic)
- **TD3** (Twin Delayed DDPG)

## Résultats de Convergence

![Convergence PPO Seeds](images/convergence_ppo_seeds0_1.png)

![SAC Convergence AAPL](images/sac_convergence_AAPL_seed42_regen.png)

![TD3 Convergence AAPL](images/td3_convergence_aapl_seed42_regen.png)

## Structure du Projet

```
ProjetRL/
 streamlit_app.py              Interface utilisateur
 config.yaml                   Configuration
 requirements.txt              Dépendances Python
 README.md                     Documentation
 
 src/                          Code source
  - environment/               Environnement RL
  - models/                    Modèles financiers
  - utils/                     Utilitaires
  
 models/                       Modèles pré-entraînés
 configs/                      Hyperparamètres
 data/                         Données
```
- `__init__.py`
- `rl_agents.py` (350 lignes) - 3 Algorithmes RL
  - PPO (Proximal Policy Optimization)
  - SAC (Soft Actor-Critic)
  - TD3 (Twin Delayed DDPG)
- `baselines.py` (450 lignes) - 4 Politiques de Benchmark
  - Target Leverage
  - Pecking Order
  - Market Timing
  - Dynamic Trade-off

### 3. **Scripts Principaux** (3 fichiers)

- `train.py` (150 lignes) - Entraînement des agents
  - Support de tous les algorithmes
  - Options pour scénarios et hyperparamètres
  - Sauvegarde et monitoring

- `evaluate.py` (250 lignes) - Évaluation et comparaison
  - Comparaison agents vs baselines
  - Génération de graphiques
  - Export des résultats en CSV

- `demo.py` (200 lignes) - Démonstration rapide
  - 4 démos indépendantes
  - Test de tous les modules
  - Pas d'entraînement GPU requis

### 4. **Documentation** (1 notebook + 2 docs)

- `notebooks/01_exploration.ipynb` (Jupyter)
  - 14 cellules interactives
  - Exploration du modèle et environnement
  - Comparaison des baselines
  - Visualisations

- Fichiers supplémentaires documentés:
  - `extract_pdf.py` (pour extraction du PDF original)

### 5. **Répertoires de Données et Logs**

```
 data/              # Données téléchargées (yfinance, FRED)
 logs/              # Logs d'entraînement et TensorBoard
 models/            # Modèles sauvegardés (PPO, SAC, TD3)
 results/           # Résultats d'évaluation et graphiques
 notebooks/         # Jupyter notebooks
```

##  Fonctionnalités Implémentées

### Environnement Gymnasium
- [x] Espace d'états continu (20 dimensions)
- [x] Espace d'actions continu (3 dimensions)
- [x] Fonction de récompense multi-objectif
- [x] Contraintes réalistes (leverage, cash, interest coverage)
- [x] 5 scénarios économiques

### Modèle Économique
- [x] Simulation de cash flows (stochastique)
- [x] Gestion dynamique de la dette et equity
- [x] Calcul du WACC et valeur d'entreprise (DCF)
- [x] Rating de crédit dynamique
- [x] Détection des défauts

### Algorithmes RL
- [x] PPO (Proximal Policy Optimization)
- [x] SAC (Soft Actor-Critic)
- [x] TD3 (Twin Delayed DDPG)
- [x] Sauvegarde/Chargement des modèles
- [x] Évaluation déterministe et stochastique

### Politiques de Benchmark
- [x] Target Leverage (ratio constant)
- [x] Pecking Order (hiérarchie de financement)
- [x] Market Timing (timing optimal)
- [x] Dynamic Trade-off (équilibre bénéfices/coûts)
- [x] Évaluation comparative

### Évaluation et Monitoring
- [x] Métriques financières (valeur, leverage, coverage)
- [x] Métriques de performance (reward, cumulative)
- [x] Robustesse multi-scénarios
- [x] Génération de graphiques
- [x] Export de résultats (CSV)

##  Configuration

### Paramètres Environnement
```yaml
ENVIRONMENT:
  initial_cash_flow: 100      # M
  initial_debt: 200           # M
  initial_equity: 300         # M
  max_leverage: 0.75
  min_interest_coverage: 2.0
```

### Poids de Récompense
```yaml
REWARD:
  alpha: 0.6   # Valeur d'entreprise
  beta: 0.2    # Flexibilité financière
  gamma: 0.15  # Coûts de détresse
  delta: 0.05  # Coûts de transaction
```

### Hyperparamètres RL
```yaml
PPO:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  clip_range: 0.2

SAC:
  learning_rate: 3e-4
  buffer_size: 1000000
  batch_size: 256
  tau: 0.005

TD3:
  learning_rate: 1e-3
  batch_size: 256
  policy_delay: 2
```

##  Utilisation

### Démonstration Rapide
```bash
python demo.py
```

### Entraînement
```bash
# PPO
python train.py --algorithm PPO --timesteps 500000

# Tous les algorithmes
python train.py --algorithm all --timesteps 500000

# Avec scénario économique
python train.py --algorithm SAC --scenario recession
```

### Évaluation
```bash
# Comparer tous les agents
python evaluate.py --scenario baseline --episodes 10

# Tous les scénarios
python evaluate.py --scenario all --episodes 20
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/01_exploration.ipynb
```

##  Métriques Disponibles

### Performance Financière
- Valeur d'entreprise (DCF)
- Rendement total des actionnaires (TSR)
- Volatilité de la valeur

### Efficience
- WACC moyen
- Leverage final
- Interest coverage moyen
- Taux de défaut

### Robustesse
- Performance multi-régimes
- Sensibilité aux paramètres
- Stabilité de la politique

##  Extensibilité

Le code est conçu pour être facilement extensible:

### Ajouter un algorithme RL
1. Créer une classe héritant de `RLAgent`
2. Implémenter `train()` et `predict()`
3. Ajouter la configuration YAML

### Ajouter une politique de benchmark
1. Créer une classe héritant de `BaselinePolicy`
2. Implémenter `get_action()`
3. Utiliser `evaluate_baseline()`

### Ajouter un scénario
1. Ajouter une section dans `config.yaml`
2. Utiliser lors de la création d'env: `make_capital_structure_env(..., scenario='custom')`

##  Théories Financières Implémentées

1. **Modigliani-Miller** - V_L = V_U + T_c × D
2. **Trade-off Theory** - Équilibre bénéfices fiscaux et coûts de détresse
3. **Pecking Order** - Hiérarchie de financement
4. **CAPM** - Coût des capitaux propres
5. **DCF** - Évaluation par flux de trésorerie actualisés

##  Dépendances Principales

| Package | Version | Utilisation |
|---------|---------|-------------|
| gymnasium | 0.29.1 | Environnement RL |
| stable-baselines3 | 2.2.1 | Algorithmes RL |
| torch | 2.2.0 | Deep Learning |
| numpy | 1.24.3 | Calculs numériques |
| pandas | 2.0.3 | DataFrames |
| matplotlib | 3.7.2 | Visualisation |
| yfinance | 0.2.32 | Données financières |

##  Critères de Succès

- [x] Environnement RL fonctionnel
- [x] 3 algorithmes entraînables
- [x] 4 politiques de benchmark
- [x] Évaluation comparative
- [x] Documentation complète
- [x] Code modulaire et testable

##  Flux de Travail Typique

1. **Setup**
   ```bash
   pip install -r requirements.txt
   python demo.py
   ```

2. **Exploration**
   - Lancer `notebooks/01_exploration.ipynb`
   - Modifier `config.yaml` si nécessaire

3. **Entraînement**
   ```bash
   python train.py --algorithm PPO --timesteps 500000
   ```

4. **Évaluation**
   ```bash
   python evaluate.py --episodes 20
   ```

5. **Analyse**
   - Vérifier les résultats dans `results/`
   - Créer des visualisations personnalisées

##  Notes Importantes

1. **GPU**: Le code utilise CUDA si disponible, sinon CPU
2. **Temps**: Entraînement PPO ~1-2h sur RTX 3090
3. **Mémoire**: ~4GB RAM minimum, 8GB+ recommandé
4. **Configuration**: Tous les paramètres sont dans `config.yaml`

##  Apprentissage

Le projet couvre:
- Deep Reinforcement Learning (PPO, SAC, TD3)
- Finance quantitative (structure de capital, WACC, DCF)
- Développement de code modularisé et professionnel
- Évaluation statistique rigoureuse

##  Support

- Consulter `README.md` pour la documentation générale
- Consulter `INSTALLATION.md` pour l'installation
- Exécuter `python demo.py` pour déboguer
- Vérifier les logs dans `logs/training.log`

---

**Projet créé**: Novembre 2024
**Version**: 0.1.0
**Statut**:  Complet et Fonctionnel
