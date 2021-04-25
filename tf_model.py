"""
    Tensorflow Keras model. (FCN)
"""

import tensorflow as tf
import tensorflow.keras as keras


class FCNLayer(keras.layers.Layer):
    def __init__(self, hyper_params):
        super(FCNLayer, self).__init__()

        self.hyper_params = hyper_params
        self.middle_layers = hyper_params['middle_layers']

        self.maxout_conv = keras.layers.Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            data_format='channels_first'
        )

        self.middle_conv_list = []

        for layer in self.middle_layers:
            assert layer >= 1 and layer % 2 == 1

            self.middle_conv_list.append(
                keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(layer, layer),
                    strides=(1, 1),
                    padding='same',
                    data_format='channels_first'
                )
            )
            self.middle_conv_list.append(keras.layers.ReLU())

        # self.middle_conv_list = keras.Sequential(self.middle_conv_list)

        self.output_conv = keras.layers.Conv2D(
            filters=10,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            data_format='channels_first'
        )

    def call(self, inputs: tf.Tensor):
        # b x 441 x m x m
        shape = [tf.shape(inputs)[k] for k in range(4)]
        maxout: tf.Tensor = self.maxout_conv(inputs)

        # Element Wise Max Pooling
        maxout = tf.reshape(maxout, shape=[-1, 64, 2, shape[2], shape[3]])
        middle = tf.reduce_max(maxout, axis=2)

        # Middle Layers
        for layer in self.middle_conv_list:
            middle = layer(middle)

        # Output Layer -> b x 10 x m x m
        out = self.output_conv(middle)

        return out


class FCNModel(keras.Model):
    def __init__(self, hyper_params):
        super(FCNModel, self).__init__()

        self.hyper_params = hyper_params
        self.fcn_layer = FCNLayer(hyper_params)

    def call(self, inputs):
        feature = inputs[0]
        mask = inputs[1]

        feature = tf.expand_dims(feature, axis=0)
        mask = tf.expand_dims(mask, axis=0)

        out = self.fcn_layer(feature)
        return out, mask


class CustomCrossEntropy(keras.losses.Loss):
    def __init__(self, name="custom_cross_entropy"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        out = y_pred[0]
        mask = y_pred[1]

        out = keras.activations.softmax(out, axis=1)
        loss = keras.losses.sparse_categorical_crossentropy(
            y_true=y_true, y_pred=out, axis=1)

        loss = tf.reduce_mean(loss * mask)

        return loss


def custom_loss(mask):
    def loss(y_true, y_pred):
        out = keras.activations.softmax(y_pred, axis=1)
        loss_val = keras.losses.sparse_categorical_crossentropy(
            y_true=y_true, y_pred=out, axis=1)

        loss_val = tf.reduce_mean(loss_val * mask)
        return loss_val

    return loss
