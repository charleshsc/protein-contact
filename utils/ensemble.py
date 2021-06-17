import os
import numpy as np
from tqdm import tqdm

# The list of prediction output
source_dir_list = ['/home/dingyueprp/protein/predict1', '/home/dingyueprp/protein/predict2', '/home/dingyueprp/protein/predict3']
# Dir of ensemble result
target_dir = '/home/dingyueprp/protein/predict'

# Get all proteins
all_proteins = os.listdir(source_dir_list[0])

# Start ensemble
for prot in tqdm(all_proteins):
    pred_list = []
    for src in source_dir_list:
        path = os.path.join(src, prot)
        if not os.path.exists(path):
            continue
        pred_list.append(np.load(path))

    pred = np.mean(pred_list, axis=0)
    np.save(os.path.join(target_dir, prot), pred)
