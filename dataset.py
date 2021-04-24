# coding:utf8
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import tarfile
import shutil


class Protein_data(Dataset):
    # Set the train and validate data
    train_proteins = None
    valid_proteins = None

    def __init__(self, hyper_params, train=True, verbose=0):
        """
        主要目标： 获取所有图片的地址
        """
        self.dist_dir = hyper_params['label_dir']
        self.feature_dir = hyper_params['feature_dir']
        self.proteins = os.listdir(self.dist_dir)
        self.validate_ratio = hyper_params['validate_ratio']
        self.train_cnt = int((1.0-self.validate_ratio) * len(self.proteins))
        self.is_train = train
        self.verbose = verbose

        if self.train_proteins is None:
            self.train_proteins = np.random.choice(
                self.proteins, self.train_cnt)
            self.valid_proteins = list(set(self.proteins) -
                                       set(self.train_proteins))

        if train:
            self.proteins = self.train_proteins
        else:
            self.proteins = self.valid_proteins

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        if self.verbose > 0:
            if self.is_train:
                print(f'Train Data: {index} fetched.')
            else:
                print(f'Validate Data: {index} fetched.')

        prot_name = self.proteins[index]
        prot_name = prot_name[:prot_name.find('.')]
        dist = self.get_label(self.dist_dir+'/'+prot_name+'.npy')
        feature = self.get_feature(self.feature_dir+'/'+prot_name + '.npy.gz')
        mask = np.where(dist == -1, 0, 1)
        label = np.zeros(dist.shape)
        label += np.where((dist >= 4) & (dist < 6),
                          np.ones_like(label), np.zeros_like(label))
        label += np.where((dist >= 6) & (dist < 8),
                          np.ones_like(label)*2, np.zeros_like(label))
        label += np.where((dist >= 8) & (dist < 10),
                          np.ones_like(label)*3, np.zeros_like(label))
        label += np.where((dist >= 10) & (dist < 12),
                          np.ones_like(label)*4, np.zeros_like(label))
        label += np.where((dist >= 12) & (dist < 14),
                          np.ones_like(label)*5, np.zeros_like(label))
        label += np.where((dist >= 14) & (dist < 16),
                          np.ones_like(label)*6, np.zeros_like(label))
        label += np.where((dist >= 16) & (dist < 18),
                          np.ones_like(label)*7, np.zeros_like(label))
        label += np.where((dist >= 18) & (dist < 20),
                          np.ones_like(label)*8, np.zeros_like(label))
        label += np.where((dist >= 20), np.ones_like(label)
                          * 9, np.zeros_like(label))
        return torch.FloatTensor(feature), torch.LongTensor(label), torch.BoolTensor(mask)

    def __len__(self):
        return len(self.proteins)

    def get_label(self, name):
        tmp_label = np.load(name)
        return tmp_label

    @staticmethod
    def get_feature(name):
        f_name = name.replace(".npy.gz", "")
        g_file = tarfile.open(name)
        g_file.extractall(f_name)
        dir_ = os.listdir(f_name)
        tmp_feature = np.load(f_name+'/'+dir_[0])
        tmp_feature = np.transpose(tmp_feature, (2, 0, 1))
        shutil.rmtree(f_name)
        return tmp_feature
