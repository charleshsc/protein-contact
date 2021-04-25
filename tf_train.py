from utils import generate_hyper_params_str
import tensorflow as tf
import tensorflow.keras as keras
from tf_dataset import Protein_data
from tf_model import FCNLayer, FCNModel, CustomCrossEntropy, custom_loss
from tensorflow.python.framework.ops import disable_eager_execution
from tqdm import tqdm
import numpy as np
import random
import os
import logging

disable_eager_execution()


# Set CUDA Environment
os.environ['CUDA_VISIBLE_DEIVCES'] = '0'

# Hyper Parameters
hyper_params = {
    'device': 'cpu',
    # 'label_dir': '/home/dingyueprp/Data/label',
    # 'feature_dir': '/home/dingyueprp/Data/feature',
    'label_dir': '/Volumes/文件/Datasets/label',
    'feature_dir': '/Volumes/文件/Datasets/feature',
    'middle_layers': [5, 5, 5, 5, 5],
    'batch_size': 1,
    'epochs': 10,
    'validate_ratio': 0.1,
    'subset_ratio': 1.0
}


def train():
    # Define Dataset
    print('Loading...')
    train_dataset = Protein_data(hyper_params, train=True, verbose=0)

    # Define Model
    print('Compiling model...')
    feature_input = keras.Input(
        shape=(441, None, None), name='feature_input')
    mask_input = keras.Input(shape=(None, None), name='mask_input')

    fcn_output = FCNLayer(hyper_params)(feature_input)
    model = keras.Model(inputs=[feature_input, mask_input], outputs=fcn_output)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=custom_loss(mask_input)
    )

    # Fit the model
    print('Fitting...')
    model.fit(
        train_dataset,
        epochs=hyper_params['epochs'],
        verbose=1,
        use_multiprocessing=True,
        workers=6
    )


if __name__ == '__main__':
    train()
