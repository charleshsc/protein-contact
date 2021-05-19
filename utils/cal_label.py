import numpy as np
import os
from tqdm import tqdm

result = [0] * 10

for root, dirs, files in os.walk('/Volumes/文件/Datasets/label/'):
    for file in tqdm(files):
        dist = np.load(os.path.join('/Volumes/文件/Datasets/label', file))
        label = np.zeros(dist.shape)
        label += np.where((dist >= 4) & (dist < 6),
                          np.ones_like(label), np.zeros_like(label))
        label += np.where((dist >= 6) & (dist < 8),
                          np.ones_like(label)*2, np.zeros_like(label))
        label += np.where((dist >= 8) & (dist < 10),
                          np.ones_like(label)*3, np.zeros_like(label))
        label += np.where((dist >= 10) & (dist < 12),
                          np.ones_like(label)*4, np.zeros_like(label))
        label += np.where((dist >= 12) & (dist < 14),
                          np.ones_like(label)*5, np.zeros_like(label))
        label += np.where((dist >= 14) & (dist < 16),
                          np.ones_like(label)*6, np.zeros_like(label))
        label += np.where((dist >= 16) & (dist < 18),
                          np.ones_like(label)*7, np.zeros_like(label))
        label += np.where((dist >= 18) & (dist < 20),
                          np.ones_like(label)*8, np.zeros_like(label))
        label += np.where((dist >= 20), np.ones_like(label)
                          * 9, np.zeros_like(label))

        mask = np.where(dist == -1, 0, 1)

        for i in range(10):
            result[i] += np.sum((label == i) * mask)

print(result)
result_sum = np.sum(result)
result = [i / result_sum for i in result]
print(result)
