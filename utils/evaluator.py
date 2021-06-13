import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.cuda
from .dataset import Protein_data
import numpy as np
from tqdm import tqdm
from .utils import cal_top
import logging


class Evaluator:
    def __init__(self, hyper_params, dataset_key='test_dir'):
        self.hyper_params = hyper_params

        self.valid_dataset = Protein_data(hyper_params[dataset_key])
        self.valid_loader = Data.DataLoader(
            dataset=self.valid_dataset,
            batch_size=hyper_params['batch_size'],
            num_workers=hyper_params['num_workers']
        )
        self.result_history = []

    def init_history(self):
        self.result_history = []

    def evaluate(self, net: nn.Module, logger: logging.Logger = None) -> float:
        print('Evaluating...')
        logger.info('Start evaluating...')
        net.eval()

        result = np.zeros(
            [self.valid_dataset.__len__(), 2, 4], dtype=np.float32)
        short_all_result = np.zeros([self.valid_dataset.__len__(), 10], dtype=np.float32)
        short_support = np.zeros([self.valid_dataset.__len__(), 10], dtype=np.float32)

        cnt = 0
        total_step = self.valid_dataset.__len__()

        with torch.no_grad():
            for step, (feature, label, mask) in enumerate(tqdm(self.valid_loader)):
                try:
                    feature = feature.to(self.hyper_params['device'])
                    label = label.to(self.hyper_params['device'])
                    mask = mask.to(self.hyper_params['device'])

                    pred = net(feature)
                    pred = F.softmax(pred, dim=1)

                    for i in range(feature.shape[0]):
                        result[cnt], short_all_result[cnt], _, short_support[cnt], _ = cal_top(
                            label[i].cpu().numpy(), mask[i].cpu().numpy(), pred[i].detach().cpu().numpy())
                        cnt += 1

                    if step % self.hyper_params['log_freq'] == 0:
                        logger.info(f'Evaluation: {step} / {total_step}')

                    torch.cuda.empty_cache()
                except Exception as err:
                    print(err)
                    logger.error(f'L={feature.shape[2]}')
                    logger.error(err)
                    torch.cuda.empty_cache()

        # Calculate Average Result
        avg_result = np.mean(result, axis=0)
        avg_short_result = np.mean(short_all_result, axis=0)
        avg_short_support = np.mean(short_support, axis=0)
        cur_result = avg_result[:, 0]+5*avg_result[:,
                                                   1]+2*avg_result[:, 2]+3*avg_result[:, 3]
        cur_result = cur_result[0] + 2 * cur_result[1]

        print(f'Evaluation Result: {cur_result}')
        if logger is not None:
            logger.info(
                f'-----------------------EVAL---------------------\nEvaluation Result: {cur_result}')
            logger.info(
                f'T10: {avg_result[0,0]}, T5: {avg_result[0,1]}, T2: {avg_result[0,2]}, T1: {avg_result[0,3]}')
            logger.info(
                f'LT10: {avg_result[1,0]}, LT5: {avg_result[1,1]}, LT2: {avg_result[1,2]}, LT1: {avg_result[1,3]}')
            logger.info(f'Short all result: {avg_short_result}')
            logger.info(f'Short support: {avg_short_support}\n')

        self.result_history.append(cur_result)

        net.train()
        return cur_result