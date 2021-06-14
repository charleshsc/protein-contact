from matplotlib import cm
from model.AttentionModel import AttentionModel
from model.DeepModel import DeepModel
from model.ResPreModel import ResPreModel
from model.DilationModel import DilationModel
from utils.utils import generate_hyper_params_str, copy_state_dict
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.cuda
from model.LossFunc import MaskedCrossEntropy, MaskedFocalLoss
from utils.dataset import Protein_data
import torch.utils.data as Data
from utils.evaluator import Evaluator
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# Set CUDA Environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Hyper Parameters
hyper_params = {
    'sample_id': 12,
    'model': 'attention', # resnet, fcn, respre, dilation, deep, attention
    'device': 'cuda',
    'label_dir': '/home/dingyueprp/Data/label',
    'feature_dir': '/home/dingyueprp/Data/feature',
    'middle_layers': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    'residual_layers': [
        (64, 64, 1, True),
        (64, 64, 1, True),
        (64, 64, 1, True),
        (64, 96, 1, True),
        (96, 96, 1, True),
        (96, 96, 1, True),
        (96, 96, 1, True),
        (96, 128, 2, True),
        (128, 128, 2, True),
        (128, 128, 2, True),
        (128, 128, 2, True),
        (128, 128, 2, True),
        (128, 128, 2, True),
        (128, 160, 4, True),
        (160, 160, 4, True),
        (160, 160, 4, True),
        (160, 160, 2, False),
        (160, 160, 1, False)
    ],
    'batch_size': 1,
    'epochs': 60,
    'dropout_rate': 0.2,
    'validate_ratio': 0.1,
    'subset_ratio': 1.0,
    'resume' : 'run/experiment_24/best_model.pth.tar'
}


# Define Dataset
print('Loading...')
test_dataset = Protein_data(
    hyper_params=hyper_params, train=False, verbose=0)

# Define Model
if hyper_params['model'] == 'respre':
    model = ResPreModel()
    model = model.to(hyper_params['device'])
elif hyper_params['model'] == 'dilation':
    model = DilationModel(hyper_params=hyper_params)
    model = model.to(hyper_params['device'])
elif hyper_params['model'] == 'deep':
    model = DeepModel(hyper_params=hyper_params)
    model = model.to(hyper_params['device'])
elif hyper_params['model'] == 'attention':
    model = AttentionModel(hyper_params=hyper_params, return_score=True)
    model = model.to(hyper_params['device'])
else:
    raise NotImplementedError(f'Model {hyper_params["model"]} not implemented.')

# Resume
if hyper_params['resume'] is not None:
    if not os.path.isfile(hyper_params['resume']):
        raise RuntimeError("=> no checkpoint found at '{}'".format(hyper_params['resume']))
    checkpoint = torch.load(hyper_params['resume'])
    hyper_params['start_epoch'] = checkpoint['epoch']
    copy_state_dict(model.state_dict(), checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(hyper_params['resume'], checkpoint['epoch']))


# Load visualize data sample_id
print('Predicting...')
feature, label, mask = test_dataset[hyper_params['sample_id']]
feature = feature.to(hyper_params['device'])

model.eval()
with torch.no_grad():
    pred, score = model(feature.unsqueeze(0))
    pred = pred.squeeze(0).detach().cpu()
    score = score.cpu()

# Visualize label
sns.set()
label_img = (label < 3) * mask
plt.grid(False)
plt.imshow(label_img, cmap='gray')
plt.title(f'Label ID: {hyper_params["sample_id"]}')
plt.savefig('img/visualize_label_attn.png')

l = score.shape[0]
attn_map = torch.zeros([l, l], dtype=torch.float)
attn_map[:l-1, :l-1] += score[1:, 1:, 0]
attn_map[:l-1, :] += score[1:, :, 1]
attn_map[:l-1, 1:] += score[1:, :l-1, 2]
attn_map[:, :l-1] += score[:, 1:, 3]
attn_map[:, :] += score[:, :, 4]
attn_map[:, 1:] += score[:, :l-1, 5]
attn_map[1:, :l-1] += score[l-1:, 1:, 6]
attn_map[1:, :] += score[l-1:, :, 7]
attn_map[1:, 1:] += score[l-1:, :l-1, 8]
attn_map = torch.sum(attn_map, dim=0).unsqueeze(0).expand([20, -1])
plt.grid(False)
plt.imshow(attn_map)
plt.colorbar()
plt.title(f'Attention Distribution')
plt.savefig('img/visualize_attn_map.png')


plt.cla()
plt.grid(False)
label_map = torch.sum((label < 3) * mask, dim=1).float() / l
plt.title(f'Contact Distribution')
plt.imshow(label_map.unsqueeze(0).expand([20, -1]))
plt.savefig('img/visualize_attn_label.png')
