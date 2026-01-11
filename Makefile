.PHONY: help install setup demo train evaluate clean test docs format lint

# Variables
PYTHON := python
PIP := pip
POETRY := poetry

help:
	@echo "Commandes disponibles:"
	@echo "  make install       - Installer les dépendances"
	@echo "  make setup         - Setup complet (install + directories)"
	@echo "  make demo          - Exécuter la démonstration"
	@echo "  make train-ppo     - Entraîner PPO"
	@echo "  make train-sac     - Entraîner SAC"
	@echo "  make train-td3     - Entraîner TD3"
	@echo "  make train-all     - Entraîner tous les algorithmes"
	@echo "  make evaluate      - Évaluer tous les agents"
	@echo "  make test          - Exécuter les tests"
	@echo "  make format        - Formater le code (black, isort)"
	@echo "  make lint          - Vérifier le code (flake8)"
	@echo "  make clean         - Nettoyer les fichiers générés"
	@echo "  make clean-all     - Nettoyer complètement (including models)"
	@echo "  make docs          - Générer la documentation"
	@echo "  make notebook      - Lancer Jupyter notebook"

install:
	$(PIP) install -r requirements.txt

setup: install
	mkdir -p logs models results data notebooks
	@echo "✓ Setup complet"

demo:
	$(PYTHON) demo.py

train-ppo:
	$(PYTHON) train.py --algorithm PPO --timesteps 500000

train-sac:
	$(PYTHON) train.py --algorithm SAC --timesteps 500000

train-td3:
	$(PYTHON) train.py --algorithm TD3 --timesteps 500000

train-all:
	$(PYTHON) train.py --algorithm all --timesteps 500000

evaluate:
	$(PYTHON) evaluate.py --scenario baseline --episodes 10

evaluate-all-scenarios:
	$(PYTHON) evaluate.py --scenario baseline --episodes 10
	$(PYTHON) evaluate.py --scenario recession --episodes 10
	$(PYTHON) evaluate.py --scenario boom --episodes 10
	$(PYTHON) evaluate.py --scenario credit_crisis --episodes 10
	$(PYTHON) evaluate.py --scenario high_volatility --episodes 10

test:
	pytest tests/ -v --cov=src

format:
	black src/ train.py evaluate.py demo.py
	isort src/ train.py evaluate.py demo.py

lint:
	flake8 src/ train.py evaluate.py demo.py --max-line-length=120

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	rm -rf .eggs/ *.egg-info/ dist/ build/

clean-all: clean
	rm -rf models/* logs/* results/*
	@echo "✓ Nettoyage complet"

docs:
	@echo "Generating documentation..."
	# Add sphinx or mkdocs command here

notebook:
	jupyter notebook notebooks/

# Commandes de développement
dev-install:
	$(PIP) install -r requirements-dev.txt
	pre-commit install

check: lint test
	@echo "✓ All checks passed"

# Commandes utilitaires
show-config:
	$(PYTHON) -c "from src.utils import load_config; import yaml; print(yaml.dump(load_config('config.yaml'), default_flow_style=False))"

show-env:
	$(PYTHON) -c "from src.utils import get_python_environment_details; print(get_python_environment_details())"

# Default target
.DEFAULT_GOAL := help
