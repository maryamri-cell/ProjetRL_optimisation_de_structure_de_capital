"""
Utilitaires de configuration et logging
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Charge la configuration YAML
    
    Args:
        config_path: Chemin vers le fichier config.yaml
        
    Returns:
        Dictionnaire de configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure le logging
    
    Args:
        log_dir: Répertoire de logs
        log_level: Niveau de logging
        
    Returns:
        Logger configuré
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("CapitalStructureRL")
    logger.setLevel(log_level)
    
    # Handler fichier
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setLevel(log_level)
    
    # Handler console
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def ensure_directories():
    """Assure que tous les répertoires nécessaires existent"""
    directories = [
        "logs", "logs/tensorboard", "logs/wandb",
        "data", "results", "models"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Logger global
logger = setup_logging()
