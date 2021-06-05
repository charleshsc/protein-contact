from matplotlib import cm
from AttentionModel import AttentionModel
from DeepModel import DeepModel
from ResPreModel import ResPreModel
from DilationModel import DilationModel
from utils import generate_hyper_params_str, copy_state_dict
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.cuda
from model import FCNModel, ResNetModel
from LossFunc import MaskedCrossEntropy, MaskedFocalLoss
import dataset
import torch.utils.data as Data
from evaluator import Evaluator
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numba


# Set CUDA Environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Hyper Parameters
hyper_params = {
    'model': 'dilation', # resnet, fcn, respre, dilation, deep, attention
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
        # (96, 96, 1, True),
        # (96, 96, 1, True),
        # (96, 96, 1, True),
        (96, 96, 1, True),
        (96, 128, 2, True),
        (128, 128, 2, True),
        (128, 128, 2, True),
        (128, 128, 2, True),
        (128, 128, 2, True),
        (128, 128, 2, True),
        (128, 160, 4, True),
        (160, 160, 4, True),
        # (160, 160, 4, True),
        (160, 160, 4, True),
        (160, 160, 2, False),
        (160, 160, 1, False)
    ],
    'batch_size': 1,
    'epochs': 60,
    'dropout_rate': 0.2,
    'validate_ratio': 0.1,
    'subset_ratio': 1.0,
    'num_workers': 20,
    'resume': 'run/experiment_23/best_model.pth.tar'
}


# Define Model
if hyper_params['model'] == 'fcn':
    model = FCNModel(hyper_params=hyper_params).to(hyper_params['device'])
elif hyper_params['model'] == 'resnet':
    model = ResNetModel(hyper_params=hyper_params)
    model = model.to(hyper_params['device'])
elif hyper_params['model'] == 'respre':
    model = ResPreModel()
    model = model.to(hyper_params['device'])
elif hyper_params['model'] == 'dilation':
    model = DilationModel(hyper_params=hyper_params)
    model = model.to(hyper_params['device'])
elif hyper_params['model'] == 'deep':
    model = DeepModel(hyper_params=hyper_params)
    model = model.to(hyper_params['device'])
elif hyper_params['model'] == 'attention':
    model = AttentionModel(hyper_params=hyper_params)
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


# Define Dataset 
valid_dataset = dataset.Protein_data(hyper_params, train=False, verbose=0)
valid_loader = Data.DataLoader(
    dataset=valid_dataset,
    batch_size=hyper_params['batch_size'],
    num_workers=hyper_params['num_workers']
)

@numba.njit
def result_to_array(result: np.ndarray, mask: np.ndarray, l: int):
    ans = np.zeros((2, 100, 100), dtype=np.float64)

    for i in range(l):
        i_index = int(i / l * 100)
        for j in range(l):
            j_index = int(j / l * 100)
            ans[0, i_index, j_index] += mask[i, j]
            ans[1, i_index, j_index] += result[i, j] * mask[i, j]

    return ans


model.eval()
cnt = np.zeros([100, 100], dtype=np.float64)
val = np.zeros([100, 100], dtype=np.float64)

current_cnt = 0
with torch.no_grad():
    for step, (feature, label, mask) in enumerate(tqdm(valid_loader)):
        try:
            feature = feature.to(hyper_params['device'])
            label = label.to(hyper_params['device'])
            l = label.shape[1]

            pred = model(feature)
            result = (torch.argmax(pred, dim=1) == label).float().squeeze(0).detach().cpu().numpy()
            mask = mask[0] * (label[0].cpu() < 9)
            cur_ans = result_to_array(result, mask.numpy(), l)
            cnt += cur_ans[0]
            val += cur_ans[1]
            current_cnt += 1
            if current_cnt >= 100:
                break

            torch.cuda.empty_cache()
        except Exception as err:
            print(err)
            torch.cuda.empty_cache()

avg_result = val / (cnt + 1e-7)
sns.set()
plt.grid(False)
plt.imshow(avg_result, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.title(f'Loss Map of {hyper_params["model"]} Long')
plt.savefig(f'img/loss_result_{hyper_params["model"]}_long.png')

np.save(f'{hyper_params["model"]}_long.npy', avg_result)
