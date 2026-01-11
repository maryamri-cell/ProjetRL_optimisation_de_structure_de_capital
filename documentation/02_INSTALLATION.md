# Installation

## Prérequis

- Python 3.9+
- pip

## Installation Rapide

```bash
# 1. Créer environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 2. Installer dépendances
pip install -r requirements.txt
```

## Vérification

```bash
streamlit run streamlit_app.py
```

L'interface devrait s'ouvrir dans votre navigateur à `http://localhost:8501`.

# Réinstallez les dépendances
pip install -r requirements.txt --force-reinstall
```

### Issue 2: CUDA not available

```bash
# Si vous avez un GPU NVIDIA mais CUDA n'est pas détecté:
# 1. Vérifiez NVIDIA drivers: nvidia-smi
# 2. Réinstallez PyTorch avec CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

### Issue 3: Out of Memory

```bash
# Si vous avez des erreurs de mémoire:
# 1. Réduisez la taille du batch
# 2. Utilisez CPU au lieu du GPU dans config.yaml
# 3. Réduisez n_steps dans la configuration
```

### Issue 4: Gymnasium import error

```bash
# Le package s'appelle gymnasium (pas gym)
pip install gymnasium --upgrade
```

## Ressources GPU

### Configurations testées

| GPU | RAM | Max Batch Size | Recommended |
|-----|-----|---|---|
| NVIDIA RTX 3080 | 10GB | 512 |  Good |
| NVIDIA RTX 3090 | 24GB | 2048 |  Excellent |
| NVIDIA V100 | 16GB | 1024 |  Good |
| NVIDIA T4 | 16GB | 512 |  Good |
| CPU (Intel i7) | 16GB RAM | 64 |  Slow |

### Google Colab

Pour développer gratuitement sur GPU:

```bash
# Installez dans Colab
!pip install -r requirements.txt

# Clonez le repo (optionnel)
!git clone https://github.com/votre-username/capital-structure-rl.git
```

### AWS/GCP/Azure

Voir les instructions spécifiques dans `docs/cloud_setup.md`

## Mise à Jour

Pour mettre à jour les dépendances:

```bash
pip install -r requirements.txt --upgrade
```

Pour mettre à jour le projet depuis GitHub:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Désinstallation

```bash
# Désactiver l'environnement
deactivate

# Supprimer l'environnement
rm -rf venv  # macOS/Linux
rmdir venv /s /q  # Windows
```

## Prochaines Étapes

Une fois installé:

1. Lire le [README.md](README.md)
2. Exécuter `python demo.py` pour une démo
3. Consulter le notebook: `notebooks/01_exploration.ipynb`
4. Lancer l'entraînement: `python train.py --help`

## Support

- **Issues**: Ouvrir une issue sur GitHub
- **Discussions**: Utiliser les discussions GitHub
- **Email**: contact@example.com

---

**Dernière mise à jour**: November 2024
