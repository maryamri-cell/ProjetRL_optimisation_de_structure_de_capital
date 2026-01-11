import numpy as np

# Load the original (non-augmented) dataset to compare
data_orig = np.load('data/training/real_data_dataset.npy', allow_pickle=True).item()
data_aug = np.load('data/training/real_data_dataset_augmented_20.npy', allow_pickle=True).item()

# Check AAPL in both
aapl_orig = data_orig['AAPL']
aapl_aug = data_aug['AAPL']

print("Original AAPL:")
print(f"  cf_raw shape: {aapl_orig['cf_raw'].shape}")
print(f"  cf_normalized shape: {aapl_orig['cf_normalized'].shape}")
print(f"  cf_raw first 5: {aapl_orig['cf_raw'][:5]}")

print("\nAugmented AAPL:")
print(f"  cf_raw shape: {aapl_aug['cf_raw'].shape}")
print(f"  cf_normalized shape: {aapl_aug['cf_normalized'].shape}")
print(f"  cf_raw first 5: {aapl_aug['cf_raw'][:5]}")
print(f"  cf_raw unique values count: {len(np.unique(aapl_aug['cf_raw']))}")
print(f"  cf_raw all: {aapl_aug['cf_raw']}")

# Check if there's a structure showing variants
print(f"\nAugmented AAPL dict keys: {list(aapl_aug.keys())}")

# Maybe there's an 'augmented_variants' key or something?
if 'augmented_variants' in aapl_aug:
    print(f"\nFound augmented_variants: {type(aapl_aug['augmented_variants'])}")
    av = aapl_aug['augmented_variants']
    if isinstance(av, list):
        print(f"  Length: {len(av)}")
        if len(av) > 0:
            print(f"  First variant type: {type(av[0])}")
            if isinstance(av[0], dict):
                print(f"  First variant keys: {list(av[0].keys())}")
