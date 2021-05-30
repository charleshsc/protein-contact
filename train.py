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
import random
import os
import logging
import Saver


# Set CUDA Environment
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Hyper Parameters
hyper_params = {
    'model': 'dilation', # resnet, fcn, respre, dilation, deep
    'device': 'cuda',
    'label_dir': '/home/dingyueprp/Data/label',
    'feature_dir': '/home/dingyueprp/Data/feature',
    # 'label_dir': '/Volumes/文件/Datasets/label',
    # 'feature_dir': '/Volumes/文件/Datasets/feature',
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
    'log_freq': 10,
    'num_workers': 16,
    'start_epoch' : 0,
    'resume' : None,
    'ft' : False,
    'class_weight': [1.5] * 9 + [1.0],
    'loss_func': 'cross', # focal, cross
    'long_length': 25 # None or int, min length for "class_weight" mask
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
    else:
        raise NotImplementedError(f'Model {hyper_params["model"]} not implemented.')

    optimizer = torch.optim.AdamW(model.parameters())
    # train_lr_scheduler = StepLR(optimizer, 10, 0.1)
    # loss_func = nn.CrossEntropyLoss(reduction='none', weight=torch.FloatTensor(hyper_params['class_weight']))
    if hyper_params['loss_func'] == 'cross':
        loss_func = MaskedCrossEntropy(hyper_params=hyper_params)
    elif hyper_params['loss_func'] == 'focal':
        loss_func = MaskedFocalLoss(alpha=hyper_params['class_weight'], gamma=1.6)
    else:
        raise NotImplementedError(f"Loss function {hyper_params['loss_func']} not implenmented")

    print('Finished')

    best_result = 0

    # Resume
    if hyper_params['resume'] is not None:
        if not os.path.isfile(hyper_params['resume']):
            raise RuntimeError("=> no checkpoint found at '{}'".format(hyper_params['resume']))
        checkpoint = torch.load(hyper_params['resume'])
        hyper_params['start_epoch'] = checkpoint['epoch']
        copy_state_dict(model.state_dict(), checkpoint['state_dict'])
        if not hyper_params['ft']:
            copy_state_dict(optimizer.state_dict(),checkpoint['optimizer'])
        best_result = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(hyper_params['resume'], checkpoint['epoch']))
    # Define Saver
    saver = Saver.Saver()

    # Start Training
    print('Start training...')

    logger.info('Start training...')
    for epoch in range(hyper_params['start_epoch'], hyper_params['epochs']):
        with tqdm(train_loader) as t:
            total_loss = 0.0
            loss_cnt = 0

            for step, (feature, label, mask) in enumerate(t):
                try:
                    feature = feature.to(hyper_params['device'])
                    label = label.to(hyper_params['device'])
                    mask = mask.to(hyper_params['device'])

                    pred = model(feature)
                    loss = loss_func(pred, label, mask)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    loss_cnt += 1

                    if hyper_params['device'] == 'cuda':
                        torch.cuda.empty_cache()

                    t.set_description(
                        f'Epoch: {epoch}, Step: {step}, L:{feature.shape[2]}, Loss: {loss.item()}')

                    if step % hyper_params['log_freq'] == 0:
                        avg_loss = total_loss / loss_cnt
                        total_loss = 0.0
                        loss_cnt = 0
                        logger.info(
                            f'Epoch: {epoch}, Step: {step}, L:{feature.shape[2]}, Loss: {avg_loss}')
                except Exception as err:
                    print(f'L={feature.shape[2]}')
                    print(err)
                    logger.error(f'L={feature.shape[2]}')
                    logger.error(err)
                    if hyper_params['device'] == 'cuda':
                        try:
                            torch.cuda.empty_cache()
                        except Exception as err:
                            logger.error(err)

        # Evaluate
        result = evaluator.evaluate(model, logger)
        # train_lr_scheduler.step()

        if result > best_result:
            is_best = True
            best_result = result
        else:
            is_best = False
        saver.save_checkpoint(state={
            'epoch' : epoch + 1,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_pred' : best_result
        }, is_best=is_best)


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
