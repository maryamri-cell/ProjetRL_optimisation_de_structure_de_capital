# Optimisation de la Structure de Capital (ProjetRL)

Ce dépôt contient une application d'optimisation de la structure de capital d'entreprises qui utilise des algorithmes d'apprentissage par renforcement (PPO, SAC, TD3) pour proposer des décisions de financement (dettes vs capitaux propres) sur un horizon discret d'étapes.

Le projet fournit à la fois :
- un environnement d'entraînement et d'évaluation (OpenAI Gym-like) pour simuler les décisions de structure de capital;
- des modèles pré-entraînés et un UI Streamlit pour lancer des optimisations, visualiser les trajectoires et exporter les résultats.

CONTENU PRINCIPAL
- `streamlit_app.py` : interface utilisateur principale (Streamlit).
- `src/` : code source (environnement, modèles, utilitaires financiers).
- `models/` : répertoires contenant modèles entraînés (PPO, SAC, TD3).
- `configs/` : fichiers de configuration et hyperparamètres.
- `data/` : exemples de données (réelles et synthétiques).
- `logs/`, `results/` : sorties d'entraînement et résultats.
- `requirements.txt` : dépendances Python.

Objectifs
- Optimiser le ratio dette/capitaux propres pour une entreprise donnée tout en tenant compte des cash-flows, du coût du capital (WACC) et des contraintes financières.
- Fournir une interface simple pour tester différents profils d'entreprise et modèles RL.

Prérequis
- Python 3.9+ (3.10 recommandé)
- `pip` pour installer les dépendances
- Accès Internet si vous souhaitez récupérer données réelles via `yfinance`

Installation rapide

1. Créer et activer un environnement virtuel (recommandé) :

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

3. (Optionnel) Si vous utilisez GPU et des versions spécifiques de bibliothèques, adaptez `requirements.txt` en conséquence.

Utilisation

1. Lancer l'interface Streamlit :

```bash
streamlit run streamlit_app.py
```

2. Dans la barre latérale :
- Choisir un profil d'entreprise (ex. `tech_startup`, `mature_company`, `apple_like`, ...)
- Ajuster les paramètres (dettes, capitaux propres, trésorerie, cash flows)
- Choisir un modèle (`ppo`, `sac`, `td3`) et le nombre d'étapes
- Cliquer sur `Lancer l'optimisation` pour exécuter l'agent et afficher les résultats

Résultats et visualisations
- **Décision Finale** : résumé (valeur initiale / valeur finale) pour Dettes, Capitaux Propres et Ratio de Levier.
- **Tableau détaillé** : trajectoire pas-à-pas (étape, décision, dettes, capitaux, levier) et possibilité d'export CSV.
- **Graphiques** : trajectoires et changements (récompense, dettes, capitaux, levier). Certaines vues peuvent être désactivées selon la configuration actuelle du code.

Configuration
- `config.yaml` : paramètres globaux de l'environnement et optimisation.
- `configs/optimized_hyperparams.yaml` et `configs/quick_test_hyperparams.yaml` : hyperparamètres pour entrainements/quick-tests.

Architectu­re et fonctionnement
- L'environnement (`src.environment.capital_structure_env`) expose un espace d'état contenant les métriques financières (debt, equity, cash, leverage, wacc, etc.) et des actions continues représentant l'ajustement du financement.
- L'agent applique une action à chaque étape; l'environnement simule l'impact financier et retourne une récompense basée sur l'objectif (par ex. ciblage d'un levier optimal, pénalités de défaut, etc.).
- Les modèles (Stable Baselines 3) sont utilisés pour exécuter des politiques déterminées et générer trajectoires d'optimisation.

Données réelles
- Quand un profil contient un `ticker` (ex. `AAPL`), l'application peut récupérer des données via `yfinance` pour pré-remplir certains paramètres. Ce comportement peut être désactivé si vous préférez travailler uniquement avec des données d'exemple.

Tests
- Plusieurs scripts de tests sont fournis (`test_api.py`, `test_optimizer.py`) qui valident des composants principaux. Exécutez :

```bash
pytest -q
```

Bonnes pratiques de développement
- Utilisez un environnement virtuel.
- Gardez `requirements.txt` synchronisé avec les dépendances réelles.
- Lors de l'entraînement de modèles, surveillez `logs/` et `tensorboard/` si configurés.

Dépannage rapide
- Erreurs d'import : assurez-vous d'avoir activé l'environnement virtuel et installé `requirements.txt`.
- Problèmes de modèle non chargé : vérifiez les chemins dans `load_models()` et l'existence des répertoires dans `models/`.
- Graphiques vides ou inchangés : vérifier que `st.session_state.last_results` contient des résultats et que l'optimisation a bien été lancée.

Contribuer
- Lire le code dans `src/` pour comprendre l'environnement et la logique de récompense.
- Créer une branche dédiée pour les changements et proposer des Pull Requests décrivant l'objectif et la validation.

Annexes
- Fichiers clés :
  - `streamlit_app.py` — UI et orchestration
  - `src/environment/capital_structure_env.py` — logique d'environnement
  - `src/models/company.py` — modèle financier de l'entreprise
  - `configs/` — hyperparamètres

