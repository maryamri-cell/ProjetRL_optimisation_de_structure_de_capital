"""
Module pour charger et traiter les données réelles Yahoo Finance
Extrait les cash flows, volatilité et paramètres pour calibrer le modèle
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import yfinance as yf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RealDataLoader:
    """Charge et traite les données réelles d'Yahoo Finance"""
    
    def __init__(self, ticker: str = "AAPL", period: str = "5y"):
        """
        Initialise le loader
        
        Args:
            ticker: Symbole boursier (AAPL, TSLA, MSFT, etc)
            period: Période historique ('5y', '10y', '1y', etc)
        """
        self.ticker = ticker
        self.period = period
        self.data = None
        self.financials = None
        
    def download_data(self) -> pd.DataFrame:
        """Télécharge les données de prix historiques"""
        logger.info(f"Téléchargement des données pour {self.ticker} ({self.period})...")
        try:
            self.data = yf.download(self.ticker, period=self.period, progress=False, auto_adjust=True)
            
            # S'assurer que les colonnes existent
            if 'Adj Close' not in self.data.columns:
                self.data['Adj Close'] = self.data.get('Close', self.data.iloc[:, 0])
            
            logger.info(f"✓ {len(self.data)} jours de données téléchargés")
            return self.data
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement: {e}")
            raise
    
    def download_financials(self) -> Dict:
        """Télécharge les données financières de l'entreprise"""
        logger.info(f"Téléchargement des données financières pour {self.ticker}...")
        try:
            ticker_obj = yf.Ticker(self.ticker)
            self.financials = {
                'info': ticker_obj.info,
                'balance_sheet': ticker_obj.balance_sheet,
                'income_stmt': ticker_obj.income_stmt,
                'cash_flow': ticker_obj.cash_flow,
                'quarterly_financials': ticker_obj.quarterly_financials,
            }
            logger.info("✓ Données financières téléchargées")
            return self.financials
        except Exception as e:
            logger.warning(f"Données financières partiellement indisponibles: {e}")
            return {}
    
    def calculate_daily_cf(self) -> np.ndarray:
        """
        Estime les cash flows journaliers à partir des prix
        Utilise la volatilité réelle et les rendements
        
        Returns:
            Array de cash flows journaliers normalisés
        """
        if self.data is None:
            self.download_data()
        
        # Calculer les rendements journaliers
        returns = self.data['Adj Close'].pct_change().dropna()
        
        # CF journalier estimé = rendement * prix moyen normalisé
        # Normaliser pour obtenir des CF positifs (en unités d'indice)
        avg_price = self.data['Adj Close'].mean()
        cf_daily = (returns * avg_price).values
        
        # Décaler pour éviter les négatifs extrêmes (utiliser max avec 0)
        cf_daily = cf_daily - cf_daily.min() + 1  # Garantir positif
        
        logger.info(f"CF journaliers calculés: mean={cf_daily.mean():.2f}, "
                   f"std={cf_daily.std():.2f}, min={cf_daily.min():.2f}")
        
        return cf_daily
    
    def calculate_parameters(self) -> Dict[str, float]:
        """
        Calcule les paramètres économiques à partir des données réelles
        
        Returns:
            Dict avec cf_mean_growth, cf_volatility, etc.
        """
        if self.data is None:
            self.download_data()
        
        returns = self.data['Adj Close'].pct_change().dropna()
        
        # Paramètres annualisés (252 jours de trading par an)
        daily_mean = returns.mean()
        daily_vol = returns.std()
        
        cf_mean_growth = daily_mean * 252  # Annualisé
        cf_volatility = daily_vol * np.sqrt(252)  # Annualisé
        
        # Extraire des infos financières si disponibles
        info = self.financials.get('info', {}) if self.financials else {}
        
        params = {
            'cf_mean_growth': float(cf_mean_growth),
            'cf_volatility': float(cf_volatility),
            'risk_free_rate': 0.02,  # Standard
            'market_risk_premium': 0.06,  # Standard
            'beta': float(info.get('beta', 1.0)),
            'tax_rate': 0.25,  # Standard corporate tax
            'base_cost_of_debt': info.get('yield', 0.03) if 'yield' in info else 0.03,
        }
        
        logger.info(f"Paramètres calibrés pour {self.ticker}:")
        for k, v in params.items():
            logger.info(f"  {k}: {v:.4f}")
        
        return params
    
    def extract_company_metrics(self) -> Dict[str, float]:
        """
        Extrait les métriques de l'entreprise réelle
        (dette, equity, cash, cash flow)
        
        Returns:
            Dict avec initial_debt, initial_equity, initial_cash, etc.
        """
        if not self.financials:
            self.download_financials()
        
        info = self.financials.get('info', {}) if self.financials else {}
        
        # Tenter d'extraire les valeurs réelles (en milliards)
        metrics = {
            'initial_cash_flow': float(info.get('operatingCashflow', 10000000000)) / 1e9,  # Billions
            'initial_debt': float(info.get('totalDebt', 50000000000)) / 1e9,
            'initial_equity': float(info.get('marketCap', 100000000000)) / 1e9,
            'initial_cash': float(info.get('totalCash', 10000000000)) / 1e9,
        }
        
        logger.info(f"Métriques de {self.ticker}:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.2f}B")
        
        return metrics
    
    def get_full_dataset(self) -> Dict:
        """
        Retourne le dataset complet pour l'entraînement
        
        Returns:
            Dict avec données, paramètres, et métriques
        """
        # Télécharger tout
        self.download_data()
        self.download_financials()
        
        # Calculer les parameters
        params = self.calculate_parameters()
        cf_daily = self.calculate_daily_cf()
        metrics = self.extract_company_metrics()
        
        return {
            'ticker': self.ticker,
            'data': self.data,
            'cf_daily': cf_daily,
            'parameters': params,
            'metrics': metrics,
            'financials': self.financials,
        }


def prepare_training_data(
    ticker: str = "AAPL",
    period: str = "5y",
    train_test_split: float = 0.8
) -> Tuple[Dict, Dict]:
    """
    Prépare les données pour l'entraînement et le test
    
    Args:
        ticker: Symbole boursier
        period: Période historique
        train_test_split: Fraction pour l'entraînement
        
    Returns:
        Tuple (train_config, test_config)
    """
    loader = RealDataLoader(ticker, period)
    full_data = loader.get_full_dataset()
    
    # Diviser les CF en train/test
    n_total = len(full_data['cf_daily'])
    n_train = int(n_total * train_test_split)
    
    cf_train = full_data['cf_daily'][:n_train]
    cf_test = full_data['cf_daily'][n_train:]
    
    # Créer configs d'entraînement et test
    train_config = {
        'ticker': ticker,
        'data_type': 'real',
        'cf_daily': cf_train,
        'parameters': full_data['parameters'],
        'metrics': full_data['metrics'],
        'period': f"Train ({period})",
    }
    
    test_config = {
        'ticker': ticker,
        'data_type': 'real',
        'cf_daily': cf_test,
        'parameters': full_data['parameters'],
        'metrics': full_data['metrics'],
        'period': f"Test ({period})",
    }
    
    logger.info(f"Train/Test split: {n_train}/{n_total-n_train} samples")
    
    return train_config, test_config


# Liste des tickers populaires (faciles à télécharger)
POPULAR_TICKERS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'TSLA': 'Tesla Inc.',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'XOM': 'Exxon Mobil Corporation',
    'JNJ': 'Johnson & Johnson',
}


if __name__ == '__main__':
    # Test simple
    print("Test de chargement des données réelles...")
    
    loader = RealDataLoader("AAPL", period="2y")
    data = loader.get_full_dataset()
    
    print(f"\n✓ Données chargées pour {data['ticker']}")
    print(f"  CF journaliers: {len(data['cf_daily'])} jours")
    print(f"  Parameters: {data['parameters']}")
    print(f"  Metrics: {data['metrics']}")
