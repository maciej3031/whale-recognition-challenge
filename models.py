import os

import tensorflow as tf

from config import CHECKPOINTS_DIR, GPU_MEMORY_FRAC_TO_USE

# In order to not use whole GPU memory
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRAC_TO_USE
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras.backend as K
from keras import Model
from keras.layers import Input, Lambda, Dense, Dropout, Flatten, Conv2D
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetLarge
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2


class BaseSiameseNetwork:
    def __init__(self, train_conf):
        self.input_shape = train_conf.input_shape
        self.timestamp = train_conf.timestamp
        self.WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, self.timestamp, "{}.weights.hdf5".format(self.timestamp))
        self.FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, self.timestamp, "{}.history.csv".format(self.timestamp))

    def _get_siamese_model(self, feature_model):
        xa_inp = Input(shape=self.input_shape, name='ImageA_Input')
        xb_inp = Input(shape=self.input_shape, name='ImageB_Input')

        xa = feature_model(xa_inp)
        xb = feature_model(xb_inp)

        x1 = Lambda(lambda x: x[0] * x[1])([xa, xb])
        x2 = Lambda(lambda x: x[0] + x[1])([xa, xb])
        x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa, xb])
        x4 = Lambda(lambda x: K.square(x))(x3)
        x = Lambda(lambda tensors: K.stack(tensors, axis=1))([x1, x2, x3, x4])

        # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
        x = Conv2D(32, (4, 1), activation='relu', padding='valid')(x)
        x = Conv2D(1, (32, 1), activation='linear', padding='valid')(x)
        x = Flatten(name='flatten')(x)

        # Weighted sum implemented as a Dense layer.
        x = Dense(1, use_bias=True, activation='sigmoid', name='weighted-average')(x)
        siamese_net = Model([xa_inp, xb_inp], x)

        return siamese_net

    def _get_triplet_loss_siamese_model(self, feature_model):
        """
        Create Siamese Network with triplet loss
        :param feature_model: Keras model object, that will be use as an encoder
        :return: Keras model object with full Siamese Network with triplet loss
        """
        anchor_input = Input(shape=self.input_shape, name='ImageA_Input')
        positive_input = Input(shape=self.input_shape, name='ImageP_Input')
        negative_input = Input(shape=self.input_shape, name='ImageN_Input')

        encoded_a = feature_model(anchor_input)
        encoded_p = feature_model(positive_input)
        encoded_n = feature_model(negative_input)

        stack_layer = Lambda(lambda tensors: K.stack(tensors, axis=1))
        merged_vectors = stack_layer([encoded_a, encoded_p, encoded_n])

        siamese_net = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vectors)

        return siamese_net

    def _get_actual_model(self, model_input, model_output):
        """
        Create final Siamese Network Keras model
        :param model_input: Input to Keras model object
        :param model_output: Output of Keras model object
        :return: Keras model object with full Siamese Network
        """

        feature_model = Model(inputs=model_input, outputs=[model_output], name='encoder')

        # Solution for broken BatchNorm moving average during transfer learning
        for layer in feature_model.layers:
            if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
                layer.trainable = True
                K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
                K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
            else:
                layer.trainable = False

        # for layer in feature_model.layers:
        #     if layer.name[-3:] == '_bn' or layer.name[-15:-2] == 'normalization' or \
        #                     layer.name[-16:-3] == 'normalization' or layer.name[-17:-4] == 'normalization':
        #         layer.trainable = False

        print(feature_model.summary())

        return self._get_triplet_loss_siamese_model(feature_model)


class ResNet50_L2(BaseSiameseNetwork):
    MODEL_NAME = 'resnet50_l2'

    def get_model(self):
        """
        Prepare full Siamese Network
        :return: Keras model object with full Siamese Network
        """
        feature_base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape,
                                      pooling='avg')

        l2_norm_layer = Lambda(lambda tensor: K.l2_normalize(tensor, axis=1))
        model_output = l2_norm_layer(feature_base_model.output)

        return self._get_actual_model(feature_base_model.input, model_output)


class InceptionResNetV2_L2(BaseSiameseNetwork):
    MODEL_NAME = 'inception_resnet_l2'

    def get_model(self):
        """
        Prepare full Siamese Network
        :return: Keras model object with full Siamese Network
        """
        feature_base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=self.input_shape,
                                               pooling='avg')

        l2_norm_layer = Lambda(lambda tensor: K.l2_normalize(tensor, axis=1))
        model_output = l2_norm_layer(feature_base_model.output)

        return self._get_actual_model(feature_base_model.input, model_output)


class Xception_L2(BaseSiameseNetwork):
    MODEL_NAME = 'xception_l2'

    def get_model(self):
        """
        Prepare full Siamese Network
        :return: Keras model object with full Siamese Network
        """
        feature_base_model = Xception(include_top=False, weights='imagenet', input_shape=self.input_shape,
                                      pooling='avg')

        l2_norm_layer = Lambda(lambda tensor: K.l2_normalize(tensor, axis=1))
        model_output = l2_norm_layer(feature_base_model.output)

        return self._get_actual_model(feature_base_model.input, model_output)


class NASNetLarge_L2(BaseSiameseNetwork):
    MODEL_NAME = 'nasnet_l2'

    def get_model(self):
        """
        Prepare full Siamese Network
        :return: Keras model object with full Siamese Network
        """
        feature_base_model = NASNetLarge(include_top=False, weights='imagenet', input_shape=self.input_shape,
                                         pooling='avg')

        l2_norm_layer = Lambda(lambda tensor: K.l2_normalize(tensor, axis=1))
        model_output = l2_norm_layer(feature_base_model.output)

        return self._get_actual_model(feature_base_model.input, model_output)


class VGG19SiameseNetworkFC_L2(BaseSiameseNetwork):
    MODEL_NAME = 'vgg19_fc_l2'

    def get_model(self):
        feature_base_model = VGG19(include_top=False, weights='imagenet', input_shape=self.input_shape)
        x = Flatten(name='flatten')(feature_base_model.output)
        x = Dense(4096, name='fc1')(x)
        x = Dropout(0.5)(x)

        l2_norm_layer = Lambda(lambda tensor: K.l2_normalize(tensor, axis=1))
        model_output = l2_norm_layer(x)

        return self._get_actual_model(feature_base_model.input, model_output)


AVAILABLE_MODELS = {Model.MODEL_NAME: Model for Model in [VGG19SiameseNetworkFC_L2, Xception_L2,
                                                          NASNetLarge_L2, ResNet50_L2, InceptionResNetV2_L2]}
