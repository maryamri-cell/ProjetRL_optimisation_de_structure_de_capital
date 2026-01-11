"""
Prépare les données collectées pour l'entraînement
"""
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path


class DataPreparator:
    """Prépare données pour entraînement RL"""

    def __init__(self, data_dir='data/real'):
        self.data_dir = data_dir

    def load_company_data(self, ticker):
        filepath = f'{self.data_dir}/{ticker}.csv'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath, parse_dates=['date'])
        return df

    def load_macro_data(self):
        filepath = f'{self.data_dir}/macro.csv'
        if not os.path.exists(filepath):
            print("⚠️  No macro data found")
            return None

        df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
        return df

    def prepare_single_company(self, ticker):
        print(f"\n🔧 Preparing {ticker}...")

        company_df = self.load_company_data(ticker)
        macro_df = self.load_macro_data()

        if macro_df is not None:
            company_df['quarter'] = pd.PeriodIndex(company_df['date'], freq='Q')
            macro_df['quarter'] = pd.PeriodIndex(macro_df.index, freq='Q')

            merged = company_df.merge(
                macro_df,
                on='quarter',
                how='left',
                suffixes=('', '_macro')
            )
        else:
            merged = company_df.copy()

        merged = merged.fillna(method='ffill').fillna(method='bfill')

        cf_raw = merged['operating_cf'].values
        cf_normalized = cf_raw / (cf_raw[0] if cf_raw[0] != 0 else 1.0)

        # Ensure all arrays are numpy arrays (handles missing columns gracefully)
        def _arr(x, default):
            if isinstance(x, (list, tuple)):
                return np.asarray(x)
            try:
                # pandas Series or numpy array
                return np.asarray(x)
            except Exception:
                return np.asarray(default)

        prepared_data = {
            'ticker': ticker,
            'n_quarters': len(merged),
            'dates': _arr(merged['date'].astype(str), []),
            'cf_normalized': _arr(cf_normalized, []),
            'cf_raw': _arr(cf_raw, []),
            'debt': _arr(merged.get('total_debt', np.full(len(merged), np.nan)), []),
            'equity': _arr(merged.get('total_equity', np.full(len(merged), np.nan)), []),
            'cash': _arr(merged.get('cash', np.full(len(merged), np.nan)), []),
            'leverage': _arr(merged.get('leverage', np.full(len(merged), np.nan)), []),
            'cf_growth': _arr(merged.get('cf_growth', np.full(len(merged), np.nan)), []),
            'gdp_growth': _arr(merged.get('gdp_growth', np.full(len(merged), np.nan)), []),
            'inflation': _arr(merged.get('inflation', np.full(len(merged), np.nan)), []),
            'rate_10y': _arr(merged.get('rate_10y', np.full(len(merged), 2.0)), []),
            'vix': _arr(merged.get('vix', np.full(len(merged), 15.0)), []),
            'sector': self._get_sector(ticker),
            'initial_values': {
                'cf': float(cf_raw[0]) if len(cf_raw) > 0 else 0.0,
                'debt': float(merged['total_debt'].iloc[0]) if 'total_debt' in merged.columns and len(merged) > 0 else 0.0,
                'equity': float(merged['total_equity'].iloc[0]) if 'total_equity' in merged.columns and len(merged) > 0 else 0.0,
                'cash': float(merged['cash'].iloc[0]) if 'cash' in merged.columns and len(merged) > 0 else 0.0
            }
        }

        print(f"✓ {ticker}: {len(merged)} quarters prepared")

        return prepared_data

    def _get_sector(self, ticker):
        sectors = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'NVDA': 'Technology',
            'JPM': 'Finance', 'BAC': 'Finance', 'WFC': 'Finance', 'GS': 'Finance',
            'WMT': 'Consumer', 'COST': 'Consumer', 'HD': 'Consumer',
            'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy',
            'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials'
        }
        return sectors.get(ticker, 'Unknown')

    def augment_cf_data(self, cf_data, n_copies=10):
        """Crée des variations des données réelles en ajoutant du bruit gaussien"""
        print(f"    🔄 Augmenting CF data: creating {n_copies} variations...")
        augmented = [cf_data]  # Inclure l'original
        for i in range(n_copies):
            noise = np.random.normal(0, cf_data.std() * 0.1, len(cf_data))
            augmented.append(cf_data + noise)
        result = np.array(augmented)
        print(f"    ✓ Created {len(augmented)} variations: shape {result.shape}")
        return result

    def prepare_all_companies(self, augment=True, n_augment=10):
        print("="*80)
        print("PREPARING TRAINING DATA")
        if augment:
            print(f"(Data augmentation: {n_augment} variations per company)")
        print("="*80)

        metadata_path = f'{self.data_dir}/metadata.json'
        if not os.path.exists(metadata_path):
            print("❌ No metadata.json found. Run collect_real_data.py first.")
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        companies = metadata.get('companies', [])
        print(f"Found {len(companies)} companies to prepare")

        dataset = {}
        for ticker in companies:
            try:
                prepared = self.prepare_single_company(ticker)
                
                # Apply data augmentation if requested
                if augment and prepared and 'cf_normalized' in prepared:
                    print(f"  🔄 Augmenting {ticker}...")
                    augmented_cf = self.augment_cf_data(prepared['cf_normalized'], n_copies=n_augment)
                    # Expand other fields to match augmented size
                    prepared['cf_normalized'] = augmented_cf
                    prepared['cf_raw'] = np.repeat(prepared['cf_raw'], n_augment + 1, axis=0) if prepared['cf_raw'].ndim > 0 else prepared['cf_raw']
                    prepared['n_quarters'] = len(augmented_cf)
                
                dataset[ticker] = prepared
            except Exception as e:
                print(f"❌ Error preparing {ticker}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✓ Prepared {len(dataset)} companies successfully")
        if augment:
            total_quarters = sum(data['n_quarters'] for data in dataset.values())
            print(f"✓ Total quarters (with augmentation): {total_quarters}")

        return dataset

    def save_training_dataset(self, dataset, output_path='data/training/real_data_dataset.npy'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np.save(output_path, dataset)

        print(f"\n💾 Training dataset saved to {output_path}")
        print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

        summary = {
            'n_companies': len(dataset),
            'companies': list(dataset.keys()),
            'statistics': {
                ticker: {
                    'n_quarters': data['n_quarters'],
                    'sector': data['sector'],
                    'date_range': f"{data['dates'][0]} to {data['dates'][-1]}",
                    'avg_cf': float(np.mean(data['cf_raw'])),
                    'avg_leverage': float(np.mean(data['leverage']))
                }
                for ticker, data in dataset.items()
            }
        }

        summary_path = output_path.replace('.npy', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Summary saved to {summary_path}")

        return output_path


def main():
    preparator = DataPreparator(data_dir='data/real')
    dataset = preparator.prepare_all_companies()
    if dataset:
        preparator.save_training_dataset(dataset)


if __name__ == '__main__':
    main()
