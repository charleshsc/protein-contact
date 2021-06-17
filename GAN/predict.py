from operator import mod
from utils import generate_hyper_params_str, copy_state_dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.cuda
from model import Generator, Discriminator
import dataset
import torch.utils.data as Data
from evaluator import Evaluator
from tqdm import tqdm
import numpy as np
import random
import os
import logging
import Saver
from Loss import GeneratorLoss, DiscriminatorLoss
import argparse
import json
from dataset import Protein_data


# Set CUDA Environment
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
hyper_params = {
    'model': 'GAN', # resnet, fcn
    'device': device,
    'batch_size': 1,
    'epochs': 20,
    'dropout_rate': 0.3,
    'validate_ratio': 0.1,
    'subset_ratio': 1.0,
    'log_freq': 10,
    'num_workers': 8,
    'start_epoch' : 0,
    'resume' : None,
    'ft' : False,
    'G_res_blocks' : 3,
    'D_res_blocks' : 3,
    'generator_train_steps_per_epoch' : 3,
    'discriminator_train_steps_per_epoch' : 3,
    'lr' : 1e-3
}

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='选定模型', default='GAN', type=str, choices=['GAN'])
parser.add_argument('--test_dir', help='测试集位置', type=str, required=True)
parser.add_argument('--target_dir', help='预测结果位置', type=str, required=True)
parser.add_argument('--checkpoint', help='检查点位置', type=str, required=True)
args = parser.parse_args()

hyper_params['test_dir'] = args.test_dir
hyper_params['target_dir'] = args.target_dir
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

def predict(logger: logging.Logger):
    # Define Dataset
    print('Loading...')

    # Define Model
    if hyper_params['model'] == 'GAN':
        generator = Generator().to(hyper_params['device'])
        discriminator = Discriminator().to(hyper_params['device'])
    else:
        raise NotImplementedError(f'Model {hyper_params["model"]} not implemented.')

    # Resume
    if hyper_params['resume'] is not None:
        if not os.path.isfile(hyper_params['resume']):
            raise RuntimeError("=> no checkpoint found at '{}'".format(hyper_params['resume']))
        checkpoint = torch.load(hyper_params['resume'])
        hyper_params['start_epoch'] = checkpoint['epoch']
        copy_state_dict(generator.state_dict(), checkpoint['generaotr_state_dict'])
        copy_state_dict(discriminator.state_dict(), checkpoint['discriminator_state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(hyper_params['resume'], checkpoint['epoch']))

    # Generat Dataset
    dataset = Protein_data(hyper_params['test_dir'], source_type='npz', return_label=False)
    dataloader = Data.DataLoader(dataset, batch_size=hyper_params['batch_size'],
                                     num_workers=hyper_params['num_workers'])

    # Predict
    generator.eval()
    with torch.no_grad():
        for feature, index in tqdm(dataloader):
            feature = feature.to(hyper_params['device'])
            pred = generator(feature)[0].detach().cpu().numpy()
            index = index[0].item()
            prot_name = dataset.get_prot_name(index)
            target_file_path = os.path.join(hyper_params['target_dir'], prot_name + '.npy')

            np.save(target_file_path, pred)


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

    try:
        predict(logger)
    except Exception as err:
        print(err)
        logger.error(str(err))
