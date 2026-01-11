"""
Script de collecte de données réelles d'entreprises S&P 500
"""
import yfinance as yf
import pandas as pd
import numpy as np
try:
    from fredapi import Fred
except ImportError:
    Fred = None
import os
from datetime import datetime


class DataCollector:
    """Collecteur de données financières réelles"""

    def __init__(self, fred_api_key=None):
        self.fred_api_key = fred_api_key
        self.fred = Fred(api_key=fred_api_key) if (fred_api_key and Fred) else None

    def collect_company_financials(self, ticker, start_date='2015-01-01'):
        """Collecte données financières complètes d'une entreprise
        Note: yfinance retourne seulement les ~7 trimestres les plus récents pour les données financières.
        Le start_date n'affecte que les données de prix (si utilisées).
        """
        print(f"\n📊 Collecting data for {ticker}...")
        try:
            stock = yf.Ticker(ticker)

            cf = stock.quarterly_cashflow

            cf_keys = ['Operating Cash Flow', 'Total Cash From Operating Activities', 'Free Cash Flow']
            operating_cf = None
            for key in cf_keys:
                if key in cf.index:
                    operating_cf = cf.loc[key]
                    break

            if operating_cf is None:
                print(f"❌ No operating CF found for {ticker}")
                return None

            bs = stock.quarterly_balance_sheet

            total_debt = bs.loc['Total Debt'] if 'Total Debt' in bs.index else None
            total_equity = bs.loc['Stockholders Equity'] if 'Stockholders Equity' in bs.index else None
            cash = bs.loc['Cash And Cash Equivalents'] if 'Cash And Cash Equivalents' in bs.index else None
            total_assets = bs.loc['Total Assets'] if 'Total Assets' in bs.index else None

            income = stock.quarterly_financials
            revenue = income.loc['Total Revenue'] if 'Total Revenue' in income.index else None
            ebitda = income.loc['EBITDA'] if 'EBITDA' in income.index else None
            net_income = income.loc['Net Income'] if 'Net Income' in income.index else None

            # Create DataFrame by aligning all series to operating_cf dates
            dates = operating_cf.index
            df = pd.DataFrame({
                'date': dates,
                'operating_cf': operating_cf.values,
                'total_debt': [total_debt[d] if (total_debt is not None and d in total_debt.index) else np.nan for d in dates],
                'total_equity': [total_equity[d] if (total_equity is not None and d in total_equity.index) else np.nan for d in dates],
                'cash': [cash[d] if (cash is not None and d in cash.index) else np.nan for d in dates],
                'total_assets': [total_assets[d] if (total_assets is not None and d in total_assets.index) else np.nan for d in dates],
                'revenue': [revenue[d] if (revenue is not None and d in revenue.index) else np.nan for d in dates],
                'ebitda': [ebitda[d] if (ebitda is not None and d in ebitda.index) else np.nan for d in dates],
                'net_income': [net_income[d] if (net_income is not None and d in net_income.index) else np.nan for d in dates],
            })

            df = df.sort_values('date').reset_index(drop=True)
            df = df.dropna(subset=['operating_cf'])

            df['leverage'] = df['total_debt'] / (df['total_debt'] + df['total_equity'] + 1e-9)
            df['cf_growth'] = df['operating_cf'].pct_change()
            df['roa'] = df['net_income'] / (df['total_assets'] + 1e-9)

            print(f"✓ {ticker}: {len(df)} quarters collected ({df['date'].min()} to {df['date'].max()})")

            return df
        except Exception as e:
            print(f"❌ Error collecting {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def collect_macro_data(self, start_date='2015-01-01'):
        if self.fred is None:
            print("⚠️  No FRED API key provided, skipping macro data")
            return None

        print("\n📈 Collecting macro data from FRED...")
        try:
            gdp = self.fred.get_series('GDP', start_date)
            cpi = self.fred.get_series('CPIAUCSL', start_date)
            rates = self.fred.get_series('DGS10', start_date)
            vix = self.fred.get_series('VIXCLS', start_date)
            unemployment = self.fred.get_series('UNRATE', start_date)

            macro_df = pd.DataFrame({
                'gdp': gdp.resample('Q').last(),
                'gdp_growth': gdp.resample('Q').last().pct_change(4) * 100,
                'inflation': cpi.resample('Q').last().pct_change(4) * 100,
                'rate_10y': rates.resample('Q').mean(),
                'vix': vix.resample('Q').mean(),
                'unemployment': unemployment.resample('Q').mean()
            })

            print(f"✓ Macro data: {len(macro_df)} quarters")
            return macro_df
        except Exception as e:
            print(f"❌ Error collecting macro data: {e}")
            return None

    def collect_sp500_sample(self, n_companies=20):
        sp500_sample = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Consumer': ['WMT', 'COST', 'HD', 'MCD', 'NKE'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'CVS', 'ABBV'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB'],
            'Industrials': ['BA', 'CAT', 'GE', 'UPS', 'HON'],
            'Materials': ['LIN', 'APD', 'SHW'],
            'Utilities': ['NEE', 'DUK', 'SO']
        }

        all_tickers = []
        for sector, tickers in sp500_sample.items():
            all_tickers.extend(tickers)

        selected_tickers = all_tickers[:n_companies]

        print(f"📊 Collecting data for {len(selected_tickers)} companies...")
        print(f"Tickers: {selected_tickers}")

        companies_data = {}
        for ticker in selected_tickers:
            df = self.collect_company_financials(ticker)
            if df is not None and len(df) >= 4:  # yfinance returns 5-7 recent quarters
                companies_data[ticker] = df

        macro_data = self.collect_macro_data()

        return companies_data, macro_data

    def save_data(self, companies_data, macro_data, output_dir='data/real'):
        os.makedirs(output_dir, exist_ok=True)

        for ticker, df in companies_data.items():
            filepath = os.path.join(output_dir, f'{ticker}.csv')
            df.to_csv(filepath, index=False)
            print(f"💾 Saved {ticker} to {filepath}")

        if macro_data is not None:
            macro_filepath = os.path.join(output_dir, 'macro.csv')
            macro_data.to_csv(macro_filepath)
            print(f"💾 Saved macro data to {macro_filepath}")

        metadata = {
            'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_companies': len(companies_data),
            'companies': list(companies_data.keys()),
            'date_range': {
                ticker: {
                    'start': str(df['date'].min()),
                    'end': str(df['date'].max()),
                    'n_quarters': len(df)
                }
                for ticker, df in companies_data.items()
            }
        }

        import json
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ All data saved to {output_dir}/")
        print(f"✓ Total: {len(companies_data)} companies")

        return metadata


def main():
    FRED_API_KEY = os.environ.get('FRED_API_KEY', None)
    collector = DataCollector(fred_api_key=FRED_API_KEY)
    companies_data, macro_data = collector.collect_sp500_sample(n_companies=20)
    if companies_data:
        collector.save_data(companies_data, macro_data)


if __name__ == '__main__':
    main()
