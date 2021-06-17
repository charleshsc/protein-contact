# coding:utf8
import numpy as np
from sklearn.metrics import precision_score
import time


def cal_acc(label, mask, large_20, pred):
    # label，pred都是二维矩阵
    result = np.zeros([4])

    mask_tmp = np.triu(mask, 0)

    L = label.shape[0]
    effect = np.where(mask_tmp.reshape(-1) != 0)
    label = label.reshape(-1)
    label = label[effect]
    pred = pred.reshape(-1)
    pred = pred[effect]
    large_20 = large_20.reshape(-1)
    large_20 = large_20[effect]
    order = np.argsort(large_20)[::-1]

    topL_10 = int(np.ceil(L/10))
    result[0] = precision_score(
        label[order[:topL_10]], pred[order[:topL_10]], average='weighted', zero_division=0)
    topL_5 = int(np.ceil(L/5))
    result[1] = precision_score(
        label[order[:topL_5]], pred[order[:topL_5]], average='weighted', zero_division=0)
    topL_2 = int(np.ceil(L/2))
    result[2] = precision_score(
        label[order[:topL_2]], pred[order[:topL_2]], average='weighted', zero_division=0)
    topL = int(np.ceil(L))
    result[3] = precision_score(
        label[order[:topL]], pred[order[:topL]], average='weighted', zero_division=0)

    return result


def cal_top(label, mask, pred):
    '''
    # 计算八个指标
    # 第一行是所有contact上的指标，第二行是长程contact上的指标，
    # 从左到右分别是topL/10，topL/5，topL/2，topL
    input
        label: L*L
        mask: L*L
        pred: 10*L*L
    output
        acc: 2*4
    '''

    acc = np.zeros([2, 4])
    large_20 = pred[9, :, :]
    large_20 = 1 - large_20
    pred_final = np.argmax(pred, 0)
    # 计算全部contact上的指标
    trunc_mat = np.zeros(label.shape)
    nn = label.shape[0]
    for kk in range(7):
        if kk != 0:
            trunc_mat = trunc_mat + \
                np.diag(np.ones(nn - kk), kk) + np.diag(np.ones(nn - kk), -kk)
        else:
            trunc_mat = trunc_mat + np.diag(np.ones(nn - kk), kk)

    trunc_mat_tmp = np.ones(trunc_mat.shape) - trunc_mat
    mask = mask * trunc_mat_tmp
    acc[0, :] = cal_acc(label, mask, large_20, pred_final)

    trunc_mat = np.zeros(label.shape)
    nn = label.shape[0]
    for kk in range(25):
        if kk != 0:
            trunc_mat = trunc_mat + \
                np.diag(np.ones(nn - kk), kk) + np.diag(np.ones(nn - kk), -kk)
        else:
            trunc_mat = trunc_mat + np.diag(np.ones(nn - kk), kk)

    trunc_mat_tmp = np.ones(trunc_mat.shape) - trunc_mat
    mask = mask * trunc_mat_tmp
    acc[1, :] = cal_acc(label, mask, large_20, pred_final)
    return acc


def generate_hyper_params_str(hyper_params):
    time_str = time.strftime("%Y%m%d_%H%M%S")
    return time_str


def copy_state_dict(cur_state_dict, pre_state_dict, prefix=''):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue

def calc_pad(k, d):
    return int((k-1)*d/2)