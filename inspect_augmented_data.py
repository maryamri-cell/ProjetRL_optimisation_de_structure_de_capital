import numpy as np

data = np.load('data/training/real_data_dataset_augmented_20.npy', allow_pickle=True)
d = data.item()

# Check how many tickers we have and their structure
print(f'Total tickers: {len(d)}')
print(f'All keys: {sorted(d.keys())}')

# Check if tickers have multiple variants
first_ticker = 'AAPL'
aapl_data = d['AAPL']
print(f'\nAAPL data type: {type(aapl_data)}')

# Check the cf_raw structure which should represent augmented variants
cf_raw = aapl_data['cf_raw']
print(f'cf_raw shape: {cf_raw.shape}')
print(f'cf_raw dtype: {cf_raw.dtype}')
print(f'First value: {cf_raw[0]}')
print(f'First 5 values: {cf_raw[:5]}')

# Check initial values
initial_values = aapl_data.get('initial_values', None)
if initial_values is not None:
    print(f'\ninitial_values type: {type(initial_values)}')
    if isinstance(initial_values, dict):
        print(f'initial_values keys: {list(initial_values.keys())}')
        for k, v in initial_values.items():
            if isinstance(v, np.ndarray):
                print(f'  {k}: shape {v.shape}, dtype {v.dtype}')
            else:
                print(f'  {k}: {type(v)} = {v}')
