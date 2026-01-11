"""
Environnement Gymnasium pour l'optimisation de structure de capital
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Optional, Any
from collections import deque
# Support both: package import (from src.environment...) and direct script execution
try:
    # Preferred when imported as a package
    from ..models import CompanyModel, EconomicParameters
    from ..utils import calculate_financial_distress_cost, calculate_transaction_costs
except Exception:
    # Fallback for direct execution: add project root to sys.path and try absolute imports
    import os
    import sys
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    try:
        # Prefer explicit src package import
        from src.models import CompanyModel, EconomicParameters
        from src.utils import calculate_financial_distress_cost, calculate_transaction_costs
    except Exception:
        # Last resort: try importing modules directly if they are top-level
        from models import CompanyModel, EconomicParameters  # type: ignore
        from utils import calculate_financial_distress_cost, calculate_transaction_costs  # type: ignore


class CapitalStructureEnv(gym.Env):
    """
    Environnement pour optimiser la structure de capital
    
    Actions:
        - Changement de dette (issuance/repayment)
        - Changement d'equity (shares issued/buyback)
        - Paiement de dividendes
    """
    
    metadata = {
        'render_modes': ['human'],
        'render_fps': 10,
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        max_steps: int = 252,  # 1 année commerciale
        scenario: str = "baseline",
        real_cf_data: Optional[np.ndarray] = None,
        disable_reward_normalization: bool = False
    ):
        """
        Initialise l'environnement
        
        Args:
            config: Dictionnaire de configuration
            max_steps: Nombre maximum de steps par épisode
            scenario: Scénario économique ('baseline', 'recession', etc.)
            real_cf_data: Array de CF réels (si fourni, utilise données réelles)
            disable_reward_normalization: Si True, retourne les raw rewards sans normalisation
        """
        self.config = config
        self.max_steps = max_steps
        self.scenario = scenario
        self.current_step = 0
        self.real_cf_data = real_cf_data
        self.disable_reward_normalization = disable_reward_normalization

        # Environnement économique
        env_config = config['ENVIRONMENT']
        reward_config = config['REWARD']
        
        # Paramètres économiques selon le scénario
        self.economic_params = self._get_scenario_params(scenario, config)
        
        # Modèle d'entreprise (avec données réelles si fourni)
        self.company = CompanyModel(
            initial_cf=env_config['initial_cash_flow'],
            initial_debt=env_config['initial_debt'],
            initial_equity=env_config['initial_equity'],
            initial_cash=50,
            params=self.economic_params,
            real_cf_data=real_cf_data  # Nouveau paramètre
        )
        
        # Contraintes
        self.max_leverage = env_config['max_leverage']
        self.min_leverage = env_config['min_leverage']
        self.min_cash_reserve = env_config['min_cash_reserve']
        self.min_interest_coverage = env_config['min_interest_coverage']
        
        # Récompense
        self.reward_weights = {
            'alpha': reward_config['alpha'],
            'beta': reward_config['beta'],
            'gamma': reward_config['gamma'],
            'delta': reward_config['delta']
        }
        
        # Espaces d'action et d'observation
        self._setup_spaces()
        
        # Historique
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
        # NOUVEAU: Tracking pour reward normalization
        self.reward_history = deque(maxlen=1000)
        self.reward_mean = 0.0
        self.reward_std = 1.0

        # NOUVEAU: Bankruptcy recovery
        self.bankruptcy_counter = 0
        self.max_bankruptcy_attempts = 5

        # NOUVEAU: Constraint violation tracking
        self.constraint_violations = []
        
    def _get_scenario_params(self, scenario: str, config: Dict) -> EconomicParameters:
        """Retourne les paramètres économiques selon le scénario"""
        scenarios = config.get('SCENARIOS', {})
        scenario_config = scenarios.get(scenario, scenarios.get('baseline'))
        
        return EconomicParameters(
            cf_mean_growth=scenario_config.get('cf_growth_mean', 0.03),
            cf_volatility=scenario_config.get('cf_volatility', 0.15),
            risk_free_rate=0.02 + scenario_config.get('rate_shock', 0),
            market_risk_premium=0.06,
            beta=1.0,
            tax_rate=0.25,
            base_cost_of_debt=0.03 + scenario_config.get('spread_shock', 0),
        )
    
    def _setup_spaces(self):
        """Configure les espaces d'action et d'observation"""
        env_config = self.config['ENVIRONMENT']
        
        # Espace d'observation (state)
        # Variables: CF, Debt, Equity, Cash, Leverage, Interest Coverage, Cash/Debt ratio, CF Growth
        # Plus: Risk-free rate, Market data, Macro indicators
        obs_size = 20
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Espace d'action (continu)
        # [debt_change, equity_change, dividend_payout]
        # Normalisés entre [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Limites pour dénormalisation
        total_cf = self.company.cf
        self.action_limits = {
            'debt_change': [-total_cf, total_cf * 2],  # Can emit or repay
            'equity_change': [-total_cf, total_cf * 2],
            'dividend_payout': [0, total_cf * 0.5],  # Max 50% CF en dividendes
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Réinitialise l'environnement
        
        Args:
            seed: Graine pour la reproductibilité
            options: Options supplémentaires
            
        Returns:
            Observation initiale et info
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Reset compteurs
        self.current_step = 0
        self.bankruptcy_counter = 0
        self.constraint_violations = []
        
        # Réinitialiser l'économie selon le scénario
        self.economic_params = self._get_scenario_params(self.scenario, self.config)
        self.company.params = self.economic_params

        # Reset company (recreate to support real_cf_data usage)
        try:
            self.company = CompanyModel(
                initial_cf=self.config['ENVIRONMENT']['initial_cash_flow'],
                initial_debt=self.config['ENVIRONMENT']['initial_debt'],
                initial_equity=self.config['ENVIRONMENT']['initial_equity'],
                initial_cash=50,
                params=self.economic_params,
                real_cf_data=self.real_cf_data
            )
        except Exception:
            # fallback to existing reset if CompanyModel doesn't support re-init
            try:
                self.company.reset()
            except Exception:
                pass
        
        # Historique
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute une action dans l'environnement
        
        Args:
            action: Action normalisée [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # NOUVEAU: 1. Clipper actions AVANT application
        action = np.clip(action, -1.0, 1.0)

        debt_change = np.clip(action[0], -0.5, 0.5)      # ±50% max (normalized)
        equity_change = np.clip(action[1], -0.3, 0.3)    # ±30% max (normalized)
        dividend = np.clip(action[2], 0.0, 0.2)          # 0-20% max (normalized)

        # NOUVEAU: 2. Vérifier contraintes AVANT application (work in normalized space)
        # Compute tentative denormalized values to evaluate leverage constraint
        tentative_denorm = self._denormalize_action([debt_change, equity_change, dividend])
        new_debt = self.company.debt + (tentative_denorm[0])
        new_equity = self.company.equity + (tentative_denorm[1])
        new_leverage = new_debt / (new_debt + new_equity + 1e-6)

        # Si viole max_leverage, ajuster action (normalized)
        if new_leverage > self.max_leverage:
            max_debt = self.max_leverage * (new_debt + new_equity)
            debt_change = (max_debt - self.company.debt) / (self.company.debt + 1e-6)
            debt_change = np.clip(debt_change, -0.5, 0.5)
            self.constraint_violations.append(('max_leverage', new_leverage))

        # Si viole min_leverage, ajuster action
        if new_leverage < self.min_leverage:
            min_debt = self.min_leverage * (new_debt + new_equity)
            debt_change = (min_debt - self.company.debt) / (self.company.debt + 1e-6)
            debt_change = np.clip(debt_change, -0.5, 0.5)
            self.constraint_violations.append(('min_leverage', new_leverage))

        # 3. Appliquer actions: dénormaliser
        denormalized_action = self._denormalize_action([debt_change, equity_change, dividend])

        # Simuler une période
        state_dict, is_default, metrics = self.company.step(
            debt_change=denormalized_action[0],
            equity_change=denormalized_action[1],
            dividend_payout=denormalized_action[2]
        )

        # 4. Calculer reward (raw)
        raw_reward = self._calculate_reward(state_dict, metrics, denormalized_action)

        # NOUVEAU: 5. Normaliser reward
        normalized_reward = self._normalize_reward(raw_reward)

        # NOUVEAU: 6. Soft termination avec recovery
        done = self._check_done_with_recovery(is_default)

        # 7. Observer state
        observation = self._get_observation()
        info = {
            'raw_reward': raw_reward,
            'normalized_reward': normalized_reward,
            'leverage': metrics.get('leverage', getattr(self.company, 'leverage', 0)),
            'coverage': metrics.get('interest_coverage', getattr(self.company, 'interest_coverage', 0)),
            'enterprise_value': metrics.get('enterprise_value', 0),
            'is_default': is_default,
            'constraint_violations': len(self.constraint_violations),
            'bankruptcy_counter': self.bankruptcy_counter
        }

        self.current_step += 1

        terminated = bool(done) or (is_default and self.bankruptcy_counter >= self.max_bankruptcy_attempts)
        truncated = self.current_step >= self.max_steps
        
        # Early termination penalty (discourage trivial short episodes)
        early_min_steps = int(self.config.get('REWARD', {}).get('min_episode_length', 10))
        early_term_penalty = float(self.config.get('REWARD', {}).get('early_termination_penalty', 20.0))
        if terminated and self.current_step < early_min_steps:
            normalized_reward -= early_term_penalty

        # Historique
        self.episode_rewards.append(normalized_reward)
        self.episode_states.append(observation)
        self.episode_actions.append([debt_change, equity_change, dividend])

        return observation, float(normalized_reward), terminated, truncated, info
    
    def _denormalize_action(self, normalized_action: np.ndarray) -> Tuple[float, float, float]:
        """Convertit l'action de [-1, 1] aux limites réelles"""
        debt_change = self._scale_action(
            normalized_action[0],
            self.action_limits['debt_change'][0],
            self.action_limits['debt_change'][1]
        )
        
        equity_change = self._scale_action(
            normalized_action[1],
            self.action_limits['equity_change'][0],
            self.action_limits['equity_change'][1]
        )
        
        dividend_payout = self._scale_action(
            normalized_action[2],
            self.action_limits['dividend_payout'][0],
            self.action_limits['dividend_payout'][1]
        )
        
        return debt_change, equity_change, dividend_payout
    
    @staticmethod
    def _scale_action(normalized: float, min_val: float, max_val: float) -> float:
        """Convertit une action normalisée [-1, 1] à [min_val, max_val]"""
        return (normalized + 1) / 2 * (max_val - min_val) + min_val
    
    def _apply_constraints(
        self,
        debt_change: float,
        equity_change: float,
        dividend_payout: float
    ) -> Tuple[float, float, float]:
        """Applique les contraintes réalistes"""
        env_config = self.config['ENVIRONMENT']
        
        # Nouvelle dette et equity proposées
        new_debt = max(0, self.company.debt + debt_change)
        new_equity = max(1, self.company.equity + equity_change)
        new_leverage = new_debt / (new_debt + new_equity)
        
        # Contrainte 1: Leverage dans les limites
        if new_leverage > self.max_leverage:
            new_debt = self.max_leverage * (new_debt + new_equity) / (1 + self.max_leverage)
            debt_change = new_debt - self.company.debt
        
        if new_leverage < self.min_leverage:
            new_debt = self.max(0, self.min_leverage * (new_debt + new_equity) / (1 + self.min_leverage))
            debt_change = new_debt - self.company.debt
        
        # Contrainte 2: Trésorerie minimale
        max_dividend = max(0, self.company.cash - self.min_cash_reserve)
        dividend_payout = min(dividend_payout, max_dividend)
        
        # Contrainte 3: Pas de dividendes si CF négatifs
        if self.company.cf < 0:
            dividend_payout = 0
        
        return debt_change, equity_change, dividend_payout
    
    def _get_observation(self) -> np.ndarray:
        """Retourne l'observation normalisée"""
        state_dict = {
            'cf': self.company.cf,
            'debt': self.company.debt,
            'equity': self.company.equity,
            'cash': self.company.cash,
            'leverage': self.company.leverage,
            'interest_coverage': min(self.company.interest_coverage, 100),
            'cash_to_debt': self.company.cash / (self.company.debt + 1),
        }
        
        # Calculer la croissance des CF
        if len(self.company.cf_history) > 1:
            cf_growth = (self.company.cf / self.company.cf_history[-2]) - 1
        else:
            cf_growth = 0
        
        state_dict['cf_growth'] = cf_growth
        
        # Ajouter des features normalisées
        features = [
            state_dict['cf'] / self.config['ENVIRONMENT']['initial_cash_flow'],
            state_dict['debt'] / self.config['ENVIRONMENT']['initial_debt'],
            state_dict['equity'] / self.config['ENVIRONMENT']['initial_equity'],
            state_dict['cash'] / (self.config['ENVIRONMENT']['initial_cash_flow'] * 2),
            state_dict['leverage'],
            state_dict['interest_coverage'] / 10,
            state_dict['cash_to_debt'],
            state_dict['cf_growth'],
        ]
        
        # Padding pour atteindre 20 features
        while len(features) < 20:
            features.append(0.0)
        
        obs = np.array(features[:20], dtype=np.float32)
        
        # Normaliser entre -1 et 1
        obs = np.clip(obs, -5, 5) / 5
        
        return obs
    
    def _calculate_reward(
        self,
        state_dict: Dict,
        metrics: Dict,
        action: list
    ) -> float:
        """
        Fonction de reward adaptative pour episodes courts et données limitées
        - Normalise la valeur PV par l'horizon réel
        - Ajoute bonus de survie (+0.1 par step sans défaut)
        - Réduit pénalité de transaction (0.001 au lieu de 0.01)
        - Ajuste poids des composantes
        """

        # ===== HORIZON ADAPTATIF =====
        # Pour données limitées: episodes courts (~5 trimestres)
        horizon = min(len(getattr(self, 'real_cf_data', [])), 5) if hasattr(self, 'real_cf_data') else 5

        # ===== COMPOSANTE 1: VALEUR D'ENTREPRISE (NORMALISÉE PAR HORIZON) =====
        wacc = getattr(self.company, 'calculate_wacc', lambda: 0.1)()
        cf = getattr(self.company, 'cf', 0.0)
        growth = getattr(self.economic_params, 'cf_mean_growth', 0.02)

        # DCF simplifié adapté à l'horizon réel
        pv = 0.0
        for t in range(1, min(horizon + 1, 6)):
            pv += cf * (1 + growth) ** t / (1 + wacc) ** t
        
        # Terminal value seulement si horizon > 5
        if horizon > 5:
            terminal_value = cf * (1 + growth) / max(wacc - growth, 0.001)
            pv += terminal_value / (1 + wacc) ** 5
        
        # Normaliser par horizon pour compenser episodes courts
        baseline = 500.0 * horizon
        value_component = pv / baseline

        # ===== COMPOSANTE 2: FLEXIBILITÉ =====
        cash_ratio = getattr(self.company, 'cash', 0.0) / (getattr(self.company, 'initial_cash', 1.0) + 1e-6)
        cash_score = np.clip(cash_ratio, 0, 2) / 2

        leverage = getattr(self.company, 'get_leverage', lambda: getattr(self.company, 'leverage', 0.0))()
        capacity_score = max(0, (self.max_leverage - leverage) / self.max_leverage)

        coverage = getattr(self.company, 'get_interest_coverage', lambda: getattr(self.company, 'interest_coverage', 0.0))()
        coverage_score = np.clip(coverage / 5.0, 0, 1)

        flexibility_component = (cash_score + capacity_score + coverage_score) / 3

        # ===== COMPOSANTE 3: OPTIMISATION =====
        target_leverage = 0.4
        leverage_distance = abs(leverage - target_leverage)

        if leverage_distance < 0.05:
            optimization_bonus = 1.0
        elif leverage_distance < 0.10:
            optimization_bonus = 0.5
        else:
            optimization_bonus = 0.0

        wacc_bonus = 0.0
        if hasattr(self, 'prev_wacc') and self.prev_wacc is not None:
            wacc_improvement = self.prev_wacc - wacc
            wacc_bonus = 5.0 * wacc_improvement
        self.prev_wacc = wacc

        optimization_component = (optimization_bonus + np.clip(wacc_bonus, -0.5, 0.5)) / 1.5

        # ===== COMPOSANTE 4: COÛTS DE DÉTRESSE =====
        if leverage > 0.3:
            distress_cost = 0.05 * np.exp(3 * (leverage - 0.3))
        else:
            distress_cost = 0.0

        if coverage < 2.0:
            distress_cost += 0.1 * (2.0 - coverage)

        distress_component = -distress_cost

        # ===== COMPOSANTE 5: COÛTS DE TRANSACTION (RÉDUITS) =====
        # Pénalité réduite de 0.01 à 0.001 pour episodes courts
        action_magnitude = np.sum(np.abs(action)) if hasattr(np, 'sum') else sum(abs(x) for x in action)
        transaction_cost = 0.001 * action_magnitude  # Réduit: 0.001 au lieu de 0.01
        transaction_component = -transaction_cost

        # ===== COMPOSANTE 6: BONUS DE SURVIE (NOUVEAU) =====
        # Récompense le simple fait de survivre sans défaut (important pour données limitées)
        survival_bonus = 0.1
        survival_component = survival_bonus

        # ===== COMPOSITION FINALE AVEC POIDS ADAPTÉS =====
        if self._is_recession():
            weights = {
                'value': 0.30,
                'flexibility': 0.35,
                'optimization': 0.10,
                'distress': 0.15,
                'transaction': 0.03,
                'survival': 0.07
            }
        else:
            weights = {
                'value': 0.40,
                'flexibility': 0.20,
                'optimization': 0.12,
                'distress': 0.10,
                'transaction': 0.03,
                'survival': 0.15  # Bonus survie plus important hors récession
            }

        total_reward = (
            weights['value'] * value_component +
            weights['flexibility'] * flexibility_component +
            weights['optimization'] * optimization_component +
            weights['distress'] * distress_component +
            weights['transaction'] * transaction_component +
            weights['survival'] * survival_component
        )

        total_reward = np.clip(total_reward, -10, 10)

        self.last_reward_info = {
            'total': total_reward,
            'value': weights['value'] * value_component,
            'flexibility': weights['flexibility'] * flexibility_component,
            'optimization': weights['optimization'] * optimization_component,
            'distress': weights['distress'] * distress_component,
            'transaction': weights['transaction'] * transaction_component,
            'survival': weights['survival'] * survival_component,
            'leverage': leverage,
            'coverage': coverage,
            'wacc': wacc,
            'horizon': horizon
        }

        return total_reward

    def _normalize_reward(self, raw_reward: float) -> float:
        """
        Reward normalization with fixed scaling (not running mean/std)
        Preserves signal strength while maintaining stability
        
        If disable_reward_normalization=True, returns completely raw rewards (no processing)
        """
        self.reward_history.append(raw_reward)

        # If disabled, return raw reward without any processing
        if self.disable_reward_normalization:
            return float(raw_reward)
        
        # Otherwise: Fixed scaling instead of running mean/std
        # This preserves the reward signal while avoiding normalization suppression
        # Clip to prevent extreme values but don't normalize to near-zero
        normalized = np.clip(raw_reward, -10, 10)
        return float(normalized)

    def _check_done_with_recovery(self, is_default: bool) -> bool:
        """Terminaison soft avec possibilité de recovery"""
        if not is_default:
            self.bankruptcy_counter = 0
            if self.current_step >= self.max_steps:
                return True
            return False

        self.bankruptcy_counter += 1
        if self.bankruptcy_counter < self.max_bankruptcy_attempts:
            # Emergency injections
            try:
                self.company.equity += 50
                self.company.cash += 25
            except Exception:
                pass
            return False

        return True

    def _is_recession(self) -> bool:
        """Détecte si en récession"""
        try:
            if len(getattr(self.company, 'cf_history', [])) >= 4:
                recent_cf = self.company.cf_history[-4:]
                cf_trend = (recent_cf[-1] - recent_cf[0]) / (recent_cf[0] + 1e-9)
                if cf_trend < -0.05:
                    return True

            spread = getattr(self.economic_params, 'base_cost_of_debt', 0.0) - getattr(self.economic_params, 'risk_free_rate', 0.0)
            if spread > 0.03:
                return True

            if hasattr(self, 'current_gdp_growth') and getattr(self, 'current_gdp_growth') < 0:
                return True
        except Exception:
            return False

    
    def _get_info(self) -> Dict:
        """Retourne des informations supplémentaires"""
        return {
            'step': self.current_step,
            'leverage': self.company.leverage,
            'rating': self.company.rating,
            'cash': self.company.cash,
            'debt': self.company.debt,
            'equity': self.company.equity,
            'interest_coverage': self.company.interest_coverage,
        }
    
    def render(self, mode: str = "human"):
        """Affiche l'état courant"""
        print(f"\n=== Step {self.current_step} ===")
        print(f"CF: {self.company.cf:.2f}")
        print(f"Debt: {self.company.debt:.2f} | Equity: {self.company.equity:.2f}")
        print(f"Leverage: {self.company.leverage:.2%} | Rating: {self.company.rating}")
        print(f"Interest Coverage: {self.company.interest_coverage:.2f}")
        print(f"Cash: {self.company.cash:.2f}")


def make_capital_structure_env(
    config: Dict[str, Any],
    max_steps: int = 252,
    scenario: str = "baseline",
    real_cf_data: Optional[np.ndarray] = None
) -> CapitalStructureEnv:
    """
    Crée un environnement CapitalStructure
    
    Args:
        config: Configuration
        max_steps: Nombre max de steps
        scenario: Scénario économique
        real_cf_data: Données CF réelles (optionnel)
    """
    return CapitalStructureEnv(config, max_steps, scenario, real_cf_data)


if __name__ == '__main__':
    # Minimal check when running the file directly to verify imports work
    print("Running capital_structure_env.py as script")
    try:
        # quick import sanity check
        print('CompanyModel, EconomicParameters and utils imported successfully')
    except Exception as e:
        print('Import fallback failed:', e)
