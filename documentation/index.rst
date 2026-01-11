====================================
Capital Structure RL Optimizer Docs
====================================

Bienvenue! Cette documentation présente le système complet d'optimisation de structure de capital utilisant l'apprentissage par renforcement (RL).

.. toctree::
   :maxdepth: 2
   :caption: Guide Principal

   01_QUICK_START
   02_INSTALLATION
   03_PROJECT_SUMMARY

.. toctree::
   :maxdepth: 2
   :caption: Algorithmes & Entraînement

   04_OPTIMIZER_USER_GUIDE
   06_OPTIMIZER_QUICK_START
   07_SAC_TRAINING_GUIDE
   09_TRAINING_REPORT_IMPROVED

.. toctree::
   :maxdepth: 2
   :caption: Interface & Évaluation

   05_INTERFACE_GUIDE
   08_SAC_COMPLETION_REPORT
   10_TESTING_GUIDE
   11_README_TESTING

.. toctree::
   :maxdepth: 2
   :caption: Analyse & Performance

   12_REWARD_NORMALIZATION_ANALYSIS
   13_TEST_PLATEAU_ANALYSIS
   14_IMPROVEMENTS_SUMMARY

.. toctree::
   :maxdepth: 2
   :caption: Intégration & Déploiement

   15_INTEGRATION_CHECKLIST
   16_VISUAL_TESTING_GUIDE
   17_WHATS_NEW
   18_PROJECT_INDEX
   19_INTERACTIVE_SUMMARY

Contenu Rapide
==============

Démarrage rapide
   - Quick Start : 5 minutes pour lancer votre premier entraînement
   - Installation : Configuration complète de l'environnement

Algorithmes Disponibles
   - PPO (Proximal Policy Optimization)
   - SAC (Soft Actor-Critic) - Recommandé
   - TD3 (Twin Delayed DDPG)

Données
   - 20 entreprises du S&P 500 réelles
   - 220 scénarios augmentés (20 variantes par entreprise)
   - Récompense moyenne SAC: 0.867 ± 0.0018

Statistiques Clés
=================

- Modèles Entraînés: PPO, SAC, TD3
- Entreprises: 20 (AAPL, MSFT, GOOGL, etc.)
- Timesteps: jusqu'à 50k par modèle
- Convergence: Atteinte en ~10k steps
- Interface Web: Flask + HTML responsif

Ressources
==========

- Fichiers MD: 19 guides documentés
- Images: 22 visualisations (convergence, évaluation, comparaison)
- Données: CSV, JSON, modèles .zip
- Code: Scripts Python complets & réutilisables

Navigation
==========

Utilisez le menu latéral pour naviguer ou consultez la section Project Summary pour une vue d'ensemble complète.

