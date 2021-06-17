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
parser.add_argument('--checkpoint', help='检查点位置', default=None, type=str, required=False)
parser.add_argument('--train_dir', help='训练集位置', type=str, required=True)
parser.add_argument('--valid_dir', help='验证集位置', type=str, required=True)
args = parser.parse_args()

hyper_params['train_dir'] = args.train_dir
hyper_params['valid_dir'] = args.valid_dir
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

def generaotr_train(generator, discriminator, criterion, optimizer, train_loader, epoch):
    generator.train()
    discriminator.eval()
    with tqdm(train_loader) as t:
        for step, (feature, label, mask) in enumerate(t):
            try:
                feature = feature.to(hyper_params['device'])
                label = label.to(hyper_params['device'])
                mask = mask.to(hyper_params['device'])

                result = generator(feature)
                with torch.no_grad():
                    # use for the loss L_adv
                    prediction = discriminator(feature, result).detach()

                optimizer.zero_grad()
                loss = criterion(prediction, result, label, mask)
                loss.backward()

                # avoid large gradient
                nn.utils.clip_grad_norm_(generator.parameters(), 5)
                optimizer.step()

                torch.cuda.empty_cache()

                t.set_description(
                    f'Epoch: {epoch}, Step: {step}, L:{feature.shape[2]}, Loss: {loss.item()}')

                if step % hyper_params['log_freq'] == 0:
                    logger.info(
                        f'Epoch: {epoch}, Step: {step}, L:{feature.shape[2]}, Loss: {loss.item()}')
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

def discriminator_train(generator, discriminator, criterion, optimizer, train_loader, epoch):
    generator.eval()
    discriminator.train()
    with tqdm(train_loader) as t:
        for step, (feature, label, mask) in enumerate(t):
            try:
                feature = feature.to(hyper_params['device'])
                label = label.to(hyper_params['device'])
                mask = mask.to(hyper_params['device'])

                with torch.no_grad():
                    fake_label = generator(feature)

                label = F.one_hot(label, num_classes=10).permute(0, 3, 1, 2).type(torch.float)
                real_result = discriminator(feature, label)
                fake_result = discriminator(feature, fake_label)

                # make the probability of the output matrix near the true matrix
                real_loss = criterion(real_result, label, mask)
                fake_loss = criterion(fake_result, fake_label, mask)
                loss = (real_loss + fake_loss) / 2

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(discriminator.parameters(), 5)
                optimizer.step()

                torch.cuda.empty_cache()

                t.set_description(
                    f'Epoch: {epoch}, Step: {step}, L:{feature.shape[2]}, Loss: {loss.item()}')

                if step % hyper_params['log_freq'] == 0:
                    logger.info(
                        f'Epoch: {epoch}, Step: {step}, L:{feature.shape[2]}, Loss: {loss.item()}')
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


def train(logger: logging.Logger):
    # Define Dataset
    print('Loading...')
    train_dataset = dataset.Protein_data(hyper_params['train_dir'])
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=True,
        num_workers=hyper_params['num_workers']
    )

    # Define Evaluator
    evaluator = Evaluator(hyper_params, dataset_key='valid_dir')

    # Define Model
    if hyper_params['model'] == 'GAN':
        generator = Generator().to(hyper_params['device'])
        discriminator = Discriminator().to(hyper_params['device'])
        generator_optimizer = torch.optim.Adam(generator.parameters(),lr=hyper_params['lr'])
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=hyper_params['lr'])
        generator_criterion = GeneratorLoss()
        discriminator_criterion = DiscriminatorLoss()
    else:
        raise NotImplementedError(f'Model {hyper_params["model"]} not implemented.')

    print('Finished')


    best_result = 0

    # Resume
    if hyper_params['resume'] is not None:
        if not os.path.isfile(hyper_params['resume']):
            raise RuntimeError("=> no checkpoint found at '{}'".format(hyper_params['resume']))
        checkpoint = torch.load(hyper_params['resume'])
        hyper_params['start_epoch'] = checkpoint['epoch']
        copy_state_dict(generator.state_dict(), checkpoint['generaotr_state_dict'])
        copy_state_dict(discriminator.state_dict(), checkpoint['discriminator_state_dict'])
        if not hyper_params['ft']:
            copy_state_dict(generator_optimizer.state_dict(),checkpoint['generaotr_optimizer'])
            copy_state_dict(discriminator_optimizer.state_dict(), checkpoint['discriminator_optimizer'])

        best_result = checkpoint['pred_best']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(hyper_params['resume'], checkpoint['epoch']))

    # Define Saver
    saver = Saver.Saver()

    # Start Training
    print('Start training...')

    logger.info('Start training...')
    for epoch in range(hyper_params['start_epoch'], hyper_params['epochs']):
        # adverse learning with generator and discriminator training alternately
        for _ in range(hyper_params['generator_train_steps_per_epoch']):
            generaotr_train(generator,discriminator,generator_criterion,generator_optimizer,train_loader,epoch)

        for _ in range(hyper_params['discriminator_train_steps_per_epoch']):
            discriminator_train(generator,discriminator,discriminator_criterion,discriminator_optimizer,train_loader,epoch)

        # Evaluate
        result = evaluator.evaluate(generator, logger)

        # Save the checkpoint
        if result > best_result:
            is_best = True
            best_result = result
        else:
            is_best = False
        saver.save_checkpoint(state={
            'epoch' : epoch + 1,
            'generaotr_state_dict' : generator.state_dict(),
            'discriminator_state_dict' : discriminator.state_dict(),
            'generaotr_optimizer' : generator_optimizer.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
            'pred_best' : best_result
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
    logger.info(json.dumps(hyper_params))

    try:
        train(logger)
    except Exception as err:
        print(err)
        logger.error(str(err))
