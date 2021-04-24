from operator import mod
import torch
import torch.nn as nn
import torch.optim
from model import FCNModel
import dataset
import torch.utils.data as Data
from evaluator import Evaluator
from tqdm import tqdm
import numpy as np
import random
import os


# Set CUDA Environment
os.environ['CUDA_VISIBLE_DEIVCES'] = '-1'


# Hyper Parameters
hyper_params = {
    'device': 'cpu',
    'label_dir': '~/Data/label',
    'feature_dir': '~/Data/feature',
    'middle_layers': [5, 5, 5, 5, 5],
    'batch_size': 1,
    'epochs': 10,
    'validate_ratio': 0.1
}


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
    train_dataset = dataset.Protein_data(hyper_params=hyper_params, train=True)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=True,
        num_workers=1
    )

    # Define Evaluator
    evaluator = Evaluator(hyper_params)

    # Define Model
    model = FCNModel(hyper_params=hyper_params)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_func = nn.CrossEntropyLoss(reduction='none')
    print('Finished')

    # Start Training
    print('Start training...')

    for epoch in range(hyper_params['epochs']):
        with tqdm(train_loader) as t:
            for step, (feature, label, mask) in enumerate(t):
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

                t.set_description(
                    f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')

        # Evaluate
        evaluator.evaluate(model)


if __name__ == '__main__':
    train()
