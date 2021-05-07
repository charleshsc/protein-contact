from operator import mod
from utils import generate_hyper_params_str
import torch
import torch.nn as nn
import torch.optim
import torch.cuda
from model import FCNModel, ResNetModel
import dataset
import torch.utils.data as Data
from evaluator import Evaluator
from tqdm import tqdm
import numpy as np
import random
import os
import logging


# Set CUDA Environment
os.environ['CUDA_VISIBLE_DEIVCES'] = '0,1'


# Hyper Parameters
hyper_params = {
    'model': 'resnet', # resnet, fcn
    'device': 'cuda',
    'label_dir': '/home/dingyueprp/Data/label',
    'feature_dir': '/home/dingyueprp/Data/feature',
    # 'label_dir': '/Volumes/文件/Datasets/label',
    # 'feature_dir': '/Volumes/文件/Datasets/feature',
    'middle_layers': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    'residual_layers': [
        (64, 32, 7, 1, 1, False),
        (32, 32, 3, 2, 1, True),
        (32, 32, 3, 2, 1, True),
        (32, 64, 3, 2, 1, True),
        (64, 64, 3, 2, 1, True),
        (64, 64, 3, 2, 1, True),
        (64, 96, 3, 2, 1, True),
        (96, 96, 3, 2, 1, True),
        (96, 96, 3, 2, 1, True),
        (96, 96, 3, 2, 1, True),
        (96, 128, 3, 2, 2, True),
        (128, 128, 3, 2, 2, True),
        (128, 128, 3, 2, 2, True),
        (128, 128, 3, 2, 2, True),
        (128, 128, 3, 2, 2, True),
        (128, 128, 3, 2, 2, True),
        (128, 160, 3, 2, 4, True),
        (160, 160, 3, 2, 4, True),
        (160, 160, 3, 2, 4, True),
        (160, 160, 3, 2, 2, False),
        (160, 160, 3, 2, 1, False)
    ],
    'batch_size': 1,
    'epochs': 20,
    'validate_ratio': 0.1,
    'subset_ratio': 1.0,
    'log_freq': 10,
    'num_workers': 20
}
info_str = generate_hyper_params_str(hyper_params)


# Manual Seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1111)


def train(logger: logging.Logger):
    # Define Dataset
    print('Loading...')
    train_dataset = dataset.Protein_data(
        hyper_params=hyper_params, train=True, verbose=0)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=True,
        num_workers=hyper_params['num_workers']
    )

    # Define Evaluator
    evaluator = Evaluator(hyper_params)

    # Define Model
    if hyper_params['model'] == 'fcn':
        model = FCNModel(hyper_params=hyper_params).to(hyper_params['device'])
    elif hyper_params['model'] == 'resnet':
        model = ResNetModel(hyper_params=hyper_params)
        model = nn.DataParallel(model)
        model = model.to(hyper_params['device'])
    else:
        raise NotImplementedError(f'Model {hyper_params["model"]} not implemented.')

    optimizer = torch.optim.AdamW(model.parameters())
    loss_func = nn.CrossEntropyLoss(reduction='none')
    print('Finished')

    # Start Training
    print('Start training...')

    logger.info('Start training...')
    for epoch in range(hyper_params['epochs']):
        with tqdm(train_loader) as t:
            for step, (feature, label, mask) in enumerate(t):
                try:
                    t.set_description(
                        f'Calculating...')
                    feature = feature.to(hyper_params['device'])
                    label = label.to(hyper_params['device'])
                    mask = mask.to(hyper_params['device'])

                    pred = model(feature)
                    loss = loss_func(pred, label)
                    loss = loss * mask

                    loss = torch.mean(loss)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if hyper_params['device'] == 'cuda':
                        torch.cuda.empty_cache()

                    t.set_description(
                        f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')

                    if step % hyper_params['log_freq'] == 0:
                        logger.info(
                            f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')
                except Exception as err:
                    print(err)
                    logger.error(err)
                    if hyper_params['device'] == 'cuda':
                        torch.cuda.empty_cache()

        # Evaluate
        evaluator.evaluate(model, logger)


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
    if hyper_params['model'] == 'fcn':
        logger.info(str(hyper_params['middle_layers']))
    elif hyper_params['model'] == 'resnet':
        logger.info(str(hyper_params['residual_layers']))

    try:
        train(logger)
    except Exception as err:
        print(err)
        logger.error(str(err))
