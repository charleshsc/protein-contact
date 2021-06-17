from model.AttentionModel import AttentionModel
from model.DeepModel import DeepModel
from model.ResPreModel import ResPreModel
from model.DilationModel import DilationModel
from model.LstmModel import LstmDilationModel
from utils.utils import generate_hyper_params_str, copy_state_dict
from model.LossFunc import MaskedCrossEntropy, MaskedFocalLoss
import torch
import torch.optim
import torch.cuda
import torch.utils.data as Data
from utils.evaluator import Evaluator
from tqdm import tqdm
import numpy as np
import random
import os
import logging
import utils.Saver as Saver
import json
import argparse
from utils.dataset import Protein_data


# Set CUDA Environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Hyper Parameters
hyper_params = {
    'model': 'attention', # respre, dilation, deep, attention, lstm
    'device': 'cuda',
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
    'log_freq': 10,
    'num_workers': 16,
    'start_epoch' : 0,
    'resume' : None,
    'realvalue': False,
    'class_weight': [1.0] * 9 + [1.0],
    'loss_func': 'cross', # focal, cross
    'long_length': None # None or int, min length for "class_weight" mask
}

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='选定模型', default='attention', type=str, required=False)
parser.add_argument('--test_dir', help='测试集位置', type=str, required=True)
parser.add_argument('--checkpoint', help='检查点位置', type=str, required=True)
args = parser.parse_args()

hyper_params['test_dir'] = args.test_dir
hyper_params['model'] = args.model
hyper_params['resume'] = args.checkpoint
info_str = generate_hyper_params_str(hyper_params)

# Manual Seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1111)


def test(logger: logging.Logger):
    # Define Dataset
    print('Loading...')

    # Define Evaluator
    evaluator = Evaluator(hyper_params)

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
        model = AttentionModel(hyper_params=hyper_params)
        model = model.to(hyper_params['device'])
    elif hyper_params['model'] == 'lstm':
        model = LstmDilationModel(hyper_params=hyper_params)
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

    # Start Evaluation
    evaluator.evaluate(model, logger=logger)


if __name__ == '__main__':
    # Config logging module.
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("logs/" + info_str + ".log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(info_str)
    logger.info(json.dumps(hyper_params))

    # Start testing
    try:
        test(logger)
    except Exception as err:
        print(err)
        logger.error(str(err))
