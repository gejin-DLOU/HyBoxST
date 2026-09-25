import os
import numpy as np

process_path = "./hest1k_datasets/colorectum/processed_data"
slides = [f"ZEN{i}" for i in range(36, 50)]

total_valid = 0
for s in slides:
    raw_n = np.load(os.path.join(process_path, "spot/idx", f"{s}_idx.npy")).shape[0]
    valid_n = np.load(os.path.join(process_path, "niche/idx", f"{s}_idx.npy")).shape[0]
    total_valid += valid_n
    print(s, "raw:", raw_n, "valid:", valid_n, "drop:", raw_n - valid_n)

print("total valid:", total_valid)