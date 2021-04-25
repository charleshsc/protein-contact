# coding:utf8
import os
import os.path as osp
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import random
import tarfile
import shutil
from io import BytesIO
import time


class Protein_data(keras.utils.Sequence):
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
        self.subset_ratio = hyper_params['subset_ratio']
        self.train_cnt = int((1.0-self.validate_ratio) *
                             len(self.proteins) * self.subset_ratio)
        self.is_train = train
        self.verbose = verbose
        self.file_read_time = []
        self.file_extract_time = []

        if self.train_proteins is None:
            if self.subset_ratio < 1.0:
                all_proteins = np.random.choice(self.proteins, int(
                    len(self.proteins) * self.subset_ratio), replace=False)
            else:
                all_proteins = self.proteins
            self.train_proteins = np.random.choice(
                all_proteins, self.train_cnt, replace=False)
            self.valid_proteins = list(set(all_proteins) -
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
                print(f'Train Data: {index} fetching...')
            else:
                print(f'Validate Data: {index} fetching...')

        prot_name = self.proteins[index]
        prot_name = prot_name[:prot_name.find('.')]
        dist = self.get_label(self.dist_dir+'/'+prot_name+'.npy')
        feature = self.get_feature(
            self.feature_dir+'/'+prot_name + '.npy.gz').astype(np.float32)
        mask = np.where(dist == -1, 0, 1)
        label = np.zeros(dist.shape, dtype=np.int64)
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

        if self.verbose > 0:
            if self.is_train:
                print(f'Train Data: {index} fetched.')
            else:
                print(f'Validate Data: {index} fetched.')

        feature = feature[np.newaxis, :, :, :]
        mask = mask[np.newaxis, :, :]
        label = label[np.newaxis, :, :]

        return [feature, mask], [label]

    def __len__(self):
        return len(self.proteins)

    @staticmethod
    def get_label(name):
        tmp_label = np.load(name)
        return tmp_label

    @staticmethod
    def get_feature(name):
        f_name = name.replace(".npy.gz", "").split('/')[-1]
        g_file = tarfile.open(name)

        arrayfile = BytesIO()

        for file in g_file.getmembers():
            arrayfile.write(g_file.extractfile(file).read())

        arrayfile.seek(0)
        # dir_ = os.listdir(f_name)
        # tmp_feature = np.load(fdata)
        tmp_feature = np.load(arrayfile)
        arrayfile.close()
        tmp_feature = np.transpose(tmp_feature, (2, 0, 1))
        return tmp_feature
