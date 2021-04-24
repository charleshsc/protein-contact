from operator import mod
from utils import generate_hyper_params_str
import torch
import torch.nn as nn
import torch.optim
import torch.cuda
from model import FCNModel
import dataset
import torch.utils.data as Data
from evaluator import Evaluator
from tqdm import tqdm
import numpy as np
import random
import os
import logging


# Set CUDA Environment
os.environ['CUDA_VISIBLE_DEIVCES'] = '0'


# Hyper Parameters
hyper_params = {
    'device': 'cuda',
    'label_dir': '/home/dingyueprp/Data/label',
    'feature_dir': '/home/dingyueprp/Data/feature',
    # 'label_dir': '/Volumes/文件/Datasets/label',
    # 'feature_dir': '/Volumes/文件/Datasets/feature',
    'middle_layers': [5, 5, 5, 5, 5],
    'batch_size': 1,
    'epochs': 10,
    'validate_ratio': 0.1,
    'subset_ratio': 1.0
}
info_str = generate_hyper_params_str(hyper_params)


# Config logging module.
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("logs/" + info_str + ".log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Manual Seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1111)


def train():
    # Define Dataset
    print('Loading...')
    train_dataset = dataset.Protein_data(
        hyper_params=hyper_params, train=True, verbose=0)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=True,
        num_workers=3
    )

    # Define Evaluator
    evaluator = Evaluator(hyper_params)

    # Define Model
    model = FCNModel(hyper_params=hyper_params).to(hyper_params['device'])
    optimizer = torch.optim.AdamW(model.parameters())
    loss_func = nn.CrossEntropyLoss(reduction='none')
    print('Finished')

    # Start Training
    print('Start training...')

    logger.info('Start training...')
    for epoch in range(hyper_params['epochs']):
        with tqdm(train_loader) as t:
            for step, (feature, label, mask) in enumerate(t):
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
                logger.info(
                    f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')

        # Evaluate
        evaluator.evaluate(model, logger)


if __name__ == '__main__':
    train()
