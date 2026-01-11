"""
Script d'entraînement rigoureux avec stabilisation et multi-seed
Implémenter les meilleures pratiques pour RL continu
"""

import os
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Import from project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.capital_structure_env import CapitalStructureEnv
from src.utils.config import load_config


class StabilityMonitoringCallback(BaseCallback):
    """Callback pour monitorer la stabilité d'entraînement"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.checkpoint_interval = 50000
        
    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info['r'])
            self.episode_lengths.append(ep_info['l'])
        
        # Log stability metrics every 10k steps
        if self.num_timesteps % 10_000 == 0 and self.num_timesteps > 0:
            if len(self.episode_rewards) >= 10:
                recent_rewards = self.episode_rewards[-10:]
                reward_mean = np.mean(recent_rewards)
                reward_std = np.std(recent_rewards)
                
                if self.verbose > 0:
                    print(f"\nStep {self.num_timesteps}: "
                          f"Reward mean={reward_mean:.2f} ± {reward_std:.2f}")
        
        return True


def make_env(config: Dict, seed: int, rank: int = 0):
    """Create a monitored environment for vectorization"""
    def _init():
        env = CapitalStructureEnv(config, scenario='baseline')
        env = Monitor(env)
        # Gymnasium doesn't have env.seed(), use reset() instead
        env.reset(seed=seed + rank)
        return env
    return _init


def train_algorithm(
    algorithm: str = 'PPO',
    n_seeds: int = 5,
    hyperparams_file: str = None,
    output_dir: str = 'models',
    log_dir: str = 'logs'
):
    """
    Train a RL algorithm with multiple seeds for robustness
    
    Args:
        algorithm: 'PPO', 'SAC', or 'TD3'
        n_seeds: Number of different seeds to train
        hyperparams_file: Path to hyperparameters YAML file
        output_dir: Directory to save models
        log_dir: Directory for tensorboard logs
    """
    
    # Load config
    config = load_config('config.yaml')
    
    # Load hyperparameters
    if hyperparams_file is None:
        hyperparams_file = 'configs/optimized_hyperparams.yaml'
    
    if not os.path.exists(hyperparams_file):
        print(f"⚠️  Hyperparameters file not found: {hyperparams_file}")
        print("Using default parameters")
        hyperparams = {}
    else:
        with open(hyperparams_file, 'r') as f:
            hyperparams = yaml.safe_load(f)
    
    algo_hyperparams = hyperparams.get(algorithm, {})
    training_config = hyperparams.get('TRAINING', {})
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Store results
    seed_results = []
    
    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"Training {algorithm} - Seed {seed + 1}/{n_seeds}")
        print(f"{'='*70}\n")
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_random_seed(seed)
        
        # Create directories for this seed
        seed_output_dir = os.path.join(output_dir, f'{algorithm}_seed{seed}')
        seed_log_dir = os.path.join(log_dir, f'{algorithm}_seed{seed}')
        
        best_model_dir = os.path.join(seed_output_dir, 'best')
        checkpoint_dir = os.path.join(seed_output_dir, 'checkpoints')
        tensorboard_log_dir = os.path.join(seed_log_dir, 'tensorboard')
        
        # Check if tensorboard is available
        try:
            import tensorboard
            tensorboard_available = True
        except ImportError:
            print("⚠️  TensorBoard not installed. Continuing without logging to tensorboard.")
            tensorboard_available = False
            tensorboard_log_dir = None
        
        os.makedirs(best_model_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        if tensorboard_log_dir is not None:
            os.makedirs(tensorboard_log_dir, exist_ok=True)

        # Check progress-bar availability (tqdm + rich required by SB3 extra)
        try:
            import tqdm  # noqa: F401
            import rich  # noqa: F401
            progress_bar_available = True
        except Exception:
            print("⚠️  tqdm and rich not installed. Disabling progress bar.")
            progress_bar_available = False
        
        # Create vectorized environments
        n_envs = training_config.get('n_envs', 4)
        
        print(f"Creating {n_envs} parallel environments...")
        env = SubprocVecEnv([make_env(config, seed, i) for i in range(n_envs)])
        eval_env = DummyVecEnv([make_env(config, seed + 1000, 0)])
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=seed_log_dir,
            eval_freq=training_config.get('eval_freq', 10_000),
            deterministic=training_config.get('deterministic_eval', True),
            render=False,
            n_eval_episodes=training_config.get('n_eval_episodes', 20),
            verbose=1
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=training_config.get('checkpoint_freq', 50_000),
            save_path=checkpoint_dir,
            name_prefix=f'{algorithm}_model',
            verbose=1
        )
        
        stability_callback = StabilityMonitoringCallback(verbose=1)
        
        callbacks = [eval_callback, checkpoint_callback, stability_callback]
        
        # Initialize model
        print(f"Initializing {algorithm} model...")
        
        try:
            if algorithm == 'PPO':
                model = PPO(
                    'MlpPolicy',
                    env,
                    learning_rate=algo_hyperparams.get('learning_rate', 3e-4),
                    n_steps=algo_hyperparams.get('n_steps', 2048),
                    batch_size=algo_hyperparams.get('batch_size', 64),
                    n_epochs=algo_hyperparams.get('n_epochs', 10),
                    gamma=algo_hyperparams.get('gamma', 0.99),
                    gae_lambda=algo_hyperparams.get('gae_lambda', 0.95),
                    clip_range=algo_hyperparams.get('clip_range', 0.2),
                    ent_coef=algo_hyperparams.get('ent_coef', 0.01),
                    vf_coef=algo_hyperparams.get('vf_coef', 0.5),
                    max_grad_norm=algo_hyperparams.get('max_grad_norm', 0.5),
                    verbose=1,
                    tensorboard_log=tensorboard_log_dir if tensorboard_available else None,
                    seed=seed
                )
            
            elif algorithm == 'SAC':
                model = SAC(
                    'MlpPolicy',
                    env,
                    learning_rate=algo_hyperparams.get('learning_rate', 3e-4),
                    buffer_size=algo_hyperparams.get('buffer_size', 1_000_000),
                    learning_starts=algo_hyperparams.get('learning_starts', 10_000),
                    batch_size=algo_hyperparams.get('batch_size', 256),
                    tau=algo_hyperparams.get('tau', 0.005),
                    gamma=algo_hyperparams.get('gamma', 0.99),
                    train_freq=algo_hyperparams.get('train_freq', 1),
                    gradient_steps=algo_hyperparams.get('gradient_steps', 1),
                    ent_coef=algo_hyperparams.get('ent_coef', 'auto'),
                    verbose=1,
                    tensorboard_log=tensorboard_log_dir if tensorboard_available else None,
                    seed=seed
                )
            
            elif algorithm == 'TD3':
                model = TD3(
                    'MlpPolicy',
                    env,
                    learning_rate=algo_hyperparams.get('learning_rate', 3e-4),
                    buffer_size=algo_hyperparams.get('buffer_size', 1_000_000),
                    learning_starts=algo_hyperparams.get('learning_starts', 10_000),
                    batch_size=algo_hyperparams.get('batch_size', 256),
                    tau=algo_hyperparams.get('tau', 0.005),
                    gamma=algo_hyperparams.get('gamma', 0.99),
                    train_freq=algo_hyperparams.get('train_freq', 1),
                    gradient_steps=algo_hyperparams.get('gradient_steps', 1),
                    policy_delay=algo_hyperparams.get('policy_delay', 2),
                    target_policy_noise=algo_hyperparams.get('target_policy_noise', 0.2),
                    target_noise_clip=algo_hyperparams.get('target_noise_clip', 0.5),
                    verbose=1,
                    tensorboard_log=tensorboard_log_dir if tensorboard_available else None,
                    seed=seed
                )
            
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        
        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            env.close()
            eval_env.close()
            continue
        
        # Train
        print(f"Training for {training_config.get('total_timesteps', 2_000_000):,} timesteps...")
        
        try:
            model.learn(
                total_timesteps=training_config.get('total_timesteps', 2_000_000),
                callback=callbacks,
                progress_bar=progress_bar_available,
                tb_log_name=f"{algorithm}_seed{seed}"
            )
            
            # Save final model
            final_model_path = os.path.join(seed_output_dir, 'final_model')
            model.save(final_model_path)
            print(f"✓ Final model saved to {final_model_path}")
            
            # Save training statistics
            stats = {
                'algorithm': algorithm,
                'seed': seed,
                'total_timesteps': training_config.get('total_timesteps', 2_000_000),
                'n_episodes': len(stability_callback.episode_rewards),
                'avg_reward': float(np.mean(stability_callback.episode_rewards[-100:])) 
                             if len(stability_callback.episode_rewards) >= 100 else 0,
                'reward_std': float(np.std(stability_callback.episode_rewards[-100:])) 
                             if len(stability_callback.episode_rewards) >= 100 else 0,
            }
            
            seed_results.append(stats)
            
            print(f"✓ Training completed successfully!")
            print(f"  Average final reward: {stats['avg_reward']:.2f} ± {stats['reward_std']:.2f}")
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user")
            model.save(os.path.join(seed_output_dir, 'interrupted_model'))
        
        except Exception as e:
            print(f"❌ Error during training: {e}")
        
        finally:
            # Cleanup
            env.close()
            eval_env.close()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY - {algorithm}")
    print(f"{'='*70}\n")
    
    if seed_results:
        avg_reward = np.mean([r['avg_reward'] for r in seed_results])
        std_reward = np.std([r['avg_reward'] for r in seed_results])
        
        print(f"Completed {len(seed_results)} seeds")
        print(f"Average final reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"\nModels saved to: {output_dir}")
        print(f"Logs saved to: {log_dir}")
    else:
        print("❌ No successful training runs")
    
    return seed_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL algorithms for capital structure optimization')
    parser.add_argument('--algorithm', type=str, default='PPO', 
                       choices=['PPO', 'SAC', 'TD3', 'ALL'],
                       help='Algorithm to train')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of seeds to train')
    parser.add_argument('--hyperparams', type=str, default='configs/optimized_hyperparams.yaml',
                       help='Path to hyperparameters file')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory')
    
    args = parser.parse_args()
    
    algorithms = ['PPO', 'SAC', 'TD3'] if args.algorithm == 'ALL' else [args.algorithm]
    
    for algo in algorithms:
        train_algorithm(
            algorithm=algo,
            n_seeds=args.seeds,
            hyperparams_file=args.hyperparams,
            output_dir=args.output_dir,
            log_dir=args.log_dir
        )


if __name__ == '__main__':
    main()
