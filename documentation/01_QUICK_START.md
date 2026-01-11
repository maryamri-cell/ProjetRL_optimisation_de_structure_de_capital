# Démarrage Rapide

## Installation (2 minutes)

```bash
# 1. Activer environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 2. Installer dépendances
pip install -r requirements.txt
```

## Utilisation (5 minutes)

```bash
# Lancer l'interface Streamlit
streamlit run streamlit_app.py
```

Dans le navigateur :
1. Choisir un profil d'entreprise (ex. `apple_like`)
2. Sélectionner un modèle (`ppo`, `sac`, `td3`)
3. Cliquer sur `Lancer l'optimisation`
4. Voir les résultats et télécharger en CSV

## Fichiers Clés

- `streamlit_app.py` — Interface utilisateur
- `config.yaml` — Configuration
- `models/` — Modèles pré-entraînés
- `src/` — Code source
