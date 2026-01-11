# Guide d'Installation

## Prérequis

- **Python 3.10+**
- **pip** ou **conda**
- **Git** (pour cloner le repo)
- **GPU NVIDIA** (optionnel mais recommandé pour l'entraînement)

## Installation Rapide

### 1. Cloner le repository (si applicable)

```bash
git clone https://github.com/votre-username/capital-structure-rl.git
cd capital-structure-rl
```

### 2. Créer un environnement virtuel

#### Avec venv (recommandé)
```bash
python -m venv venv

# Activation (Windows)
venv\Scripts\activate

# Activation (macOS/Linux)
source venv/bin/activate
```

#### Avec conda
```bash
conda create -n capital-rl python=3.10
conda activate capital-rl
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Pour une installation plus légère (sans GPU):
```bash
pip install -r requirements-minimal.txt
```

### 4. Vérifier l'installation

```bash
python demo.py
```

Vous devriez voir les démos s'exécuter avec succès.

## Installation Avancée

### Avec GPU NVIDIA (CUDA 11.8+)

```bash
# PyTorch avec CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Puis installer les autres dépendances
pip install -r requirements.txt
```

### Setup Développeur

Si vous développez sur le projet:

```bash
# Installation en mode développeur
pip install -e .

# Ajouter les outils de développement
pip install -r requirements-dev.txt

# Setup pre-commit hooks (optionnel)
pre-commit install
```

### Avec Docker

```bash
# Build l'image
docker build -t capital-structure-rl:latest .

# Run un conteneur
docker run --gpus all -it capital-structure-rl:latest
```

## Configuration

### Variables d'Environnement

Créez un fichier `.env` basé sur `.env.example`:

```bash
cp .env.example .env
```

Éditez `.env` avec vos valeurs:

```
DEBUG=False
SEED=42
DEVICE=cuda
LOG_LEVEL=INFO
```

### Configuration YAML

Modifiez `config.yaml` pour ajuster:
- Les paramètres initiaux de l'entreprise
- Les hyperparamètres RL
- Les scénarios économiques

Voir le fichier pour plus de détails.

## Vérification de l'Installation

### Test minimal

```bash
python -c "import gymnasium; import stable_baselines3; import torch; print(' All imports successful!')"
```

### Test complet

```bash
python demo.py
```

Cela devrait exécuter 4 démos sans erreur.

### Test GPU (optionnel)

```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Dépannage

### Issue 1: ModuleNotFoundError

```bash
# Assurez-vous que l'environnement virtuel est activé
python -m pip install --upgrade pip

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
