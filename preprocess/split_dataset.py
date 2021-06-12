import os
import random
import shutil
from tqdm import tqdm

# Parameters
from_path = '/home/dingyueprp/Data/'
to_path = '/home/dingyueprp/Data/'
split_p = [0.8, 0.1, 0.1]

for item in ['train', 'valid', 'test']:
    path = os.path.join(to_path, item)
    feature_path = os.path.join(to_path, item, 'feature')
    label_path = os.path.join(to_path, item, 'label')

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    if not os.path.exists(label_path):
        os.mkdir(label_path)

# Read File
feature_path = os.path.join(from_path, 'feature')
label_path = os.path.join(from_path, 'label')

for root, dirs, files in tqdm(os.walk(feature_path, topdown=False)):
    for name in files:
        if name.startswith('.') or not name.endswith('npy.gz'):
            continue
        
        flag_str = ''
        p = random.random()
        if p < split_p[0]:
            flag_str = 'train'
        elif p < split_p[0] + split_p[1]:
            flag_str = 'valid'
        else:
            flag_str = 'test'

        label_name = name.split('.')[0]+'.npy'
        origin_feature_path = os.path.join(feature_path, name)
        origin_label_path = os.path.join(label_path, label_name)

        target_feature_path = os.path.join(to_path, flag_str, 'feature', name)
        target_label_path = os.path.join(to_path, flag_str, 'label', label_name)

        # Move File
        shutil.move(origin_feature_path, target_feature_path)
        shutil.move(origin_label_path, target_label_path)
