"""Helpers financiers minimalistes utilisés par l'environnement.

Fournit des implémentations simples et robustes pour :
- calculate_wacc
- calculate_interest_coverage
- calculate_credit_spread
- calculate_financial_distress_cost
- calculate_transaction_costs
- calculate_enterprise_value
- normalize_state
- denormalize_action

Ces fonctions sont volontairement conservatrices et bien documentées
pour faciliter les tests et l'intégration.
"""
from typing import Iterable, Optional
import numpy as np


def calculate_wacc(equity: float, debt: float, cost_of_equity: float, cost_of_debt: float, tax_rate: float = 0.25) -> float:
    """Calcule un WACC simple pondéré.

    Args:
        equity: valeur des fonds propres
        debt: valeur de la dette
        cost_of_equity: coût des capitaux propres (en fraction, ex. 0.08)
        cost_of_debt: coût de la dette (en fraction, ex. 0.03)
        tax_rate: taux d'imposition

    Retourne:
        WACC (float)
    """
    total = max(equity + debt, 1e-9)
    we = equity / total
    wd = debt / total
    after_tax_debt = cost_of_debt * (1.0 - tax_rate)
    return we * cost_of_equity + wd * after_tax_debt


def calculate_interest_coverage(ebit: float, interest_expense: float) -> float:
    """Ratio d'intérêt couvert : EBIT / intérêts. Gère division par zéro."""
    if interest_expense == 0:
        return float('inf') if ebit > 0 else 0.0
    return ebit / interest_expense


def calculate_credit_spread(leverage: float, base_spread: float = 0.01) -> float:
    """Approximation du spread de crédit en fonction du leverage.

    Simple fonction croissante.
    """
    lev = max(0.0, float(leverage))
    return base_spread + 0.02 * lev


def calculate_financial_distress_cost(leverage: float) -> float:
    """Coût de détresse financier approximatif (croissance exponentielle au-delà d'un seuil).

    Empêche valeurs négatives et renvoie 0 si leverage faible.
    """
    lev = max(0.0, float(leverage))
    if lev <= 0.3:
        return 0.0
    return 0.05 * float(np.exp(3.0 * (lev - 0.3)))


def calculate_transaction_costs(debt_change: float, equity_change: float, debt: float, equity: float) -> float:
    """Estimation simple des coûts de transaction (en unités monétaires).

    - Coût proportionnel à la taille des changements par rapport à l'enterprise scale.
    - Renvoie une valeur absolue (mêmes unités que `debt`/`equity`).
    """
    scale = max(debt + equity, 1.0)
    debt_cost_pct = 0.005  # 0.5% pour dette
    equity_cost_pct = 0.01  # 1% pour equity

    cost = debt_cost_pct * abs(debt_change) + equity_cost_pct * abs(equity_change)
    # Optionnel: limiter coût à une fraction raisonnable
    return float(cost)


def calculate_enterprise_value(cf: float, wacc: float, growth: float, years: int = 5) -> float:
    """DCF simplifié: PV des CF pour `years` + valeur terminale.

    - Protège contre wacc <= growth en ajoutant une petite marge.
    """
    cf = float(cf)
    growth = float(growth)
    wacc = float(wacc)
    if wacc <= growth:
        wacc = growth + 1e-3

    pv = 0.0
    for t in range(1, years + 1):
        pv += cf * (1 + growth) ** t / (1 + wacc) ** t

    terminal = cf * (1 + growth) / max(wacc - growth, 1e-6)
    pv += terminal / (1 + wacc) ** years
    return float(pv)


def normalize_state(state: Iterable[float], mean: Optional[Iterable[float]] = None, std: Optional[Iterable[float]] = None) -> np.ndarray:
    """Normalise un vecteur d'état (Z-score)."""
    arr = np.asarray(state, dtype=float)
    if mean is None or std is None:
        mu = arr.mean()
        sigma = arr.std() + 1e-8
        return (arr - mu) / sigma
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(std, dtype=float)
    return (arr - mu) / (sigma + 1e-8)


def denormalize_action(normalized: Iterable[float], min_val: Iterable[float], max_val: Iterable[float]) -> np.ndarray:
    """Dénormalise action(s) de [-1,1] vers [min_val, max_val]."""
    n = np.asarray(normalized, dtype=float)
    min_a = np.asarray(min_val, dtype=float)
    max_a = np.asarray(max_val, dtype=float)
    return (n + 1.0) / 2.0 * (max_a - min_a) + min_a


__all__ = [
    'calculate_wacc',
    'calculate_interest_coverage',
    'calculate_credit_spread',
    'calculate_financial_distress_cost',
    'calculate_transaction_costs',
    'calculate_enterprise_value',
    'normalize_state',
    'denormalize_action',
]
