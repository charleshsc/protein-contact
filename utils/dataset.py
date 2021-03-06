# coding:utf8
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tarfile
from io import BytesIO


class Protein_data(Dataset):
    def __init__(self, dataset_dir, realvalue, source_type='tar', return_label=True):
        """
        主要目标： 获取所有图片的地址
        dataset_dir: 数据集的文件夹，分为feature与label两部分
        source_type: tar（npy.gz），npz（npz）
        return_label: True（返回特征、标签与mask）、False（返回特征、index）
        """
        self.feature_dir = os.path.join(dataset_dir, 'feature')
        if return_label:
            self.label_dir = os.path.join(dataset_dir, 'label')
        self.proteins = os.listdir(self.feature_dir)
        self.source_type = source_type
        self.return_label = return_label
        self.realvalue = realvalue

    def get_prot_name(self, index):
        """
        根据index返回蛋白质名称
        """
        return self.proteins[index].split('.')[0]

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        prot_name = self.proteins[index]
        prot_name = prot_name[:prot_name.find('.')]
        if self.return_label:
            dist = self.get_label(self.label_dir+'/'+prot_name+'.npy')
        feature = None
        while feature is None:
            try:
                feature = self.get_feature(self.feature_dir+'/'+prot_name + '.npy.gz', self.source_type)
            except Exception as err:
                print(err)
        
        if self.return_label:
            if self.realvalue:
                mask = np.where(dist == -1, 0, 1)
                label = dist
                label[label < 0] = 0
                label[label > 20] = 20

                return torch.FloatTensor(feature), torch.LongTensor(label), torch.BoolTensor(mask)
            else:
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
        else:
            return torch.FloatTensor(feature), torch.tensor(index).long()

    def __len__(self):
        return len(self.proteins)

    def get_label(self, name):
        tmp_label = np.load(name)
        return tmp_label

    @staticmethod
    def get_feature(name, source_type='tar'):
        """
        返回特征
        source_type: tar或npz。其中，tar使用内存解压方法加速。
        """
        if source_type == 'tar':
            g_file = tarfile.open(name)

            arrayfile = BytesIO()

            for file in g_file.getmembers():
                arrayfile.write(g_file.extractfile(file).read())

            arrayfile.seek(0)
            tmp_feature = np.load(arrayfile)
            arrayfile.close()
            tmp_feature = np.transpose(tmp_feature, (2, 0, 1))
            return tmp_feature
        elif source_type == 'npz':
            tmp_feature = np.load(name.replace('npy.gz', 'npz'))['a']
            tmp_feature = np.transpose(tmp_feature, (2, 0, 1))
            return tmp_feature
        else:
            raise NotImplementedError(f"Source type {source_type} not implemented.")
