"""Prioritized Experience Replay Buffer for SAC

Implémente un replay buffer qui prioritise les expériences importantes
basé sur l'erreur TD (Temporal Difference error).
"""
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Tuple, Optional


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer qui prioritise les expériences avec erreur TD élevée"""
    
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            buffer_size: Taille max du buffer
            observation_space: Espace d'observation
            action_space: Espace d'action
            device: Device torch
            n_envs: Nombre d'environnements
            optimize_memory_usage: Optimiser la mémoire
            alpha: Exponent de prioritisation (0=uniform, 1=full prioritization)
            beta: Exponent d'importance sampling (0=no correction, 1=full correction)
            epsilon: Small constant pour éviter priorités nulles
        """
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage
        )
        
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
        # Priorités initiales (max priority)
        self.priorities = np.ones(self.buffer_size, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos,
    ) -> None:
        """Ajoute une expérience avec priorité maximale"""
        # Stocker l'index avant d'ajouter
        idx = self.pos
        
        # Appeler parent pour ajouter l'expérience
        super().add(obs, next_obs, action, reward, done, infos)
        
        # Assigner priorité maximale à la nouvelle expérience
        self.priorities[idx] = self.max_priority
    
    def sample(
        self,
        batch_size: int,
        env: Optional[object] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample un batch d'expériences selon les priorités
        
        Returns:
            batch, importance_weights, indices
        """
        # Obtenir la taille valide du buffer
        valid_size = min(self.pos, self.buffer_size) if self.pos < self.buffer_size else self.buffer_size
        if valid_size == 0:
            # Si buffer vide, utiliser un sample aléatoire simple
            return super().sample(batch_size, env)
        
        valid_priorities = self.priorities[:valid_size]
        
        # Probabilités selon priorités
        probs = valid_priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices selon probabilités
        indices = np.random.choice(valid_size, size=batch_size, p=probs, replace=True)
        
        # Calculer importance sampling weights
        weights = (valid_size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normaliser par le max
        
        # Récupérer les données via la méthode parente
        data = self._get_samples(indices, env)
        
        return data, weights, indices
    
    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """
        Met à jour les priorités basées sur l'erreur TD
        
        Args:
            indices: Indices des expériences
            td_errors: Valeurs absolues des erreurs TD
        """
        # Priorités = |TD error| + epsilon
        priorities = np.abs(td_errors) + self.epsilon
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


class SAC_Prioritized:
    """
    Wrapper pour SAC avec Prioritized Replay Buffer
    
    Utilise les erreurs TD pour mettre à jour les priorités.
    Note: Ceci est un mixin à utiliser avec un SAC créé par stable-baselines3
    """
    
    @staticmethod
    def update_priorities_from_td_error(model, indices, td_errors):
        """Mise à jour les priorités du buffer après un training step
        
        Args:
            model: Modèle SAC stable-baselines3
            indices: Indices des expériences entraînées
            td_errors: Erreurs TD absolues
        """
        if hasattr(model.replay_buffer, 'update_priorities'):
            model.replay_buffer.update_priorities(indices, td_errors)
