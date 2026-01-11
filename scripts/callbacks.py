"""
Callbacks personnalisés pour l'entraînement
"""
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class LearningRateScheduler(BaseCallback):
    """
    Réduit progressivement le learning rate
    """
    def __init__(self, initial_lr=3e-4, verbose=0):
        super().__init__(verbose)
        self.initial_lr = initial_lr

    def _on_step(self) -> bool:
        if self.num_timesteps == 500_000:
            new_lr = self.initial_lr * 0.5
            try:
                self.model.lr_schedule = lambda _: new_lr
            except Exception:
                pass
            if self.verbose:
                print(f"\n📉 Reducing learning rate to {new_lr}")

        if self.num_timesteps == 1_000_000:
            new_lr = self.initial_lr * 0.1
            try:
                self.model.lr_schedule = lambda _: new_lr
            except Exception:
                pass
            if self.verbose:
                print(f"\n📉 Reducing learning rate to {new_lr}")

        if self.num_timesteps == 1_500_000:
            new_lr = self.initial_lr * 0.05
            try:
                self.model.lr_schedule = lambda _: new_lr
            except Exception:
                pass
            if self.verbose:
                print(f"\n📉 Reducing learning rate to {new_lr}")

        return True


class DetailedMonitoringCallback(BaseCallback):
    """
    Monitore métriques détaillées pendant entraînement
    """
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            try:
                # training_env may be a VecEnv
                env = self.training_env
                if hasattr(env, 'get_attr'):
                    companies = env.get_attr('company')
                    leverages = [getattr(c, 'get_leverage', lambda: getattr(c, 'leverage', 0.0))() for c in companies]
                    coverages = [getattr(c, 'get_interest_coverage', lambda: getattr(c, 'interest_coverage', 0.0))() for c in companies]
                    waccs = [getattr(c, 'calculate_wacc', lambda: 0.0)() for c in companies]

                    self.logger.record('metrics/avg_leverage', np.mean(leverages))
                    self.logger.record('metrics/avg_coverage', np.mean(np.clip(coverages, 0, 100)))
                    self.logger.record('metrics/avg_wacc', np.mean(waccs))
            except Exception:
                pass

        return True


class StabilityMonitoringCallback(BaseCallback):
    """
    Monitore stabilité de l'entraînement
    """
    def __init__(self, check_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                try:
                    self.rewards.append(ep_info['r'])
                except Exception:
                    pass

        if self.num_timesteps % self.check_freq == 0 and len(self.rewards) > 100:
            recent_rewards = self.rewards[-100:]
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            cv = std_reward / (abs(mean_reward) + 1e-8)

            self.logger.record('stability/mean_reward', mean_reward)
            self.logger.record('stability/std_reward', std_reward)
            self.logger.record('stability/cv', cv)

            if cv > 2.0 and self.verbose:
                print(f"\n⚠️  High variance detected: CV={cv:.2f}")

        return True
