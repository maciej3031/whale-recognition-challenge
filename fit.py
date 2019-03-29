import argparse
import csv
import gc
import os
import warnings
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import Adam

from config import MAX_TRAIN_STEPS, DEFAULT_LEARNING_RATE, DEFAULT_LOAD_WEIGHTS, MAX_TRAIN_EPOCHS, INPUT_SHAPE, \
    DEFAULT_FREEZE_LAYERS, DEFAULT_MODEL_NAME, DEFAULT_MARGIN, BATCH_SIZE, VALID_IMG_NUM, CONFIG_HISTORY_FILE, \
    DEFAULT_TRAINING_CONFIG, CHECKPOINTS_DIR, DEFAULT_HARD_SAMPLING_BATCH, DEFAULT_TIMESTAMP, DEFAULT_USE_HARD_BATCH, \
    USE_SIAMESE_MODEL
from models import AVAILABLE_MODELS
from utils.data_generators import TripletDataGenerator, PairDataGenerator
from utils.losses_and_metrics import triplet_loss, triplet_acc


def warn(*args, **kwargs):
    pass


warnings.warn = warn
gc.enable()  # memory is tight


class TrainingConfigurator:
    def __init__(self, kwargs):
        self.model_name = kwargs['model_name']
        self.learning_rate = kwargs['learning_rate']
        self.load_weights = kwargs['load_weights']  # to be removed
        self.freeze_layers = kwargs['freeze_layers']
        self.margin = kwargs['margin']
        self.hard_sampling_batch_size = kwargs['hard_sampling_batch_size']
        self.batch_size = kwargs['batch_size']
        self.number_of_validation_imgs = kwargs['number_of_validation_imgs']
        self.input_shape = kwargs['input_shape']
        self.model = None
        self.keras_model = None
        self.encoder = None
        self.timestamp = kwargs['timestamp']
        self.hard_batch_approach = DEFAULT_USE_HARD_BATCH
        self.siamese_model = USE_SIAMESE_MODEL

        if self.timestamp is None:
            self._prepare_directory()

        self._load_models()
        self._prepare_models()

    def _prepare_directory(self):
        self.timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        os.mkdir(os.path.join(CHECKPOINTS_DIR, self.timestamp))

    def _load_models(self):
        """
        Prepare instance of SiameseNetwork and Keras models: full siamese and encoders
        """
        ModelClass = AVAILABLE_MODELS.get(self.model_name)
        self.model = ModelClass(self)
        self.keras_model = self.model.get_model()
        self.encoder = self.keras_model.get_layer('encoder')

    def _prepare_models(self):
        """
        Prepare Keras models basing on training configuration (Freeze layers, load weights)
        """
        if self.freeze_layers is not None:
            self._set_freeze_layers()
        self._load_weight_if_possible()
        print(self.keras_model.summary())
        self.show_configuration()

    def _load_weight_if_possible(self):
        """
        Load weights for Keras Siamese model if they exist
        """
        try:
            self.keras_model.load_weights(self.model.WEIGHT_PATH)
            print('Weights loaded!')
        except OSError:
            print('No file with weights available! Starting from scratch...')

    def _set_freeze_layers(self):
        """
        Make some number of first few layers non-trainable. Number is defined by self.freeze_layers parameter.
        """
        for layer in self.encoder.layers[:self.freeze_layers]:
            layer.trainable = False

    def show_configuration(self):
        """
        Show configuration
        """
        keys = self.get_configuration_parameters_names()
        data = self.get_configuration_parameters_values()
        print('\nTRAINING CONFIGURATION:\n')
        for pos, param in enumerate(keys):
            print('{}: {}'.format(param, data[pos]))
        print('\n')

    def show_loss(self, acc=True):
        """
        Print loss history
        :param acc: optionally print accuracy history
        """
        loss_history = pd.read_csv(self.model.FIT_HISTORY_PATH, sep=';')
        epochs = loss_history.epoch
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

        _ = ax1.plot(epochs, loss_history.loss, 'b-',
                     epochs, loss_history.val_loss, 'r-')
        ax1.legend(['Training', 'Validation'])
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('Loss')
        ax1.set_title('Loss')

        if acc:
            _ = ax2.plot(epochs, loss_history.acc, 'b-',
                         epochs, loss_history.val_acc, 'r-')
            ax2.legend(['Training', 'Validation'])
            ax2.set_xlabel('epochs')
            ax2.set_ylabel('Acc')
            ax2.set_title('Acc')

    def _get_callbacks(self):
        """
        Prepare Keras callback for training
        :return: List of Keras callbacks
        """
        csv_logger = CSVLogger(self.model.FIT_HISTORY_PATH, append=False, separator=';')
        checkpoint = ModelCheckpoint(self.model.WEIGHT_PATH,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     save_weights_only=True)

        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.2,
                                           patience=1,
                                           verbose=1,
                                           mode='min',
                                           min_delta=0.0001,
                                           cooldown=0,
                                           min_lr=1e-10)

        early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=5, min_delta=0.0001)
        tb = TensorBoard(log_dir="./Graph", write_grads=True,
                         histogram_freq=1, write_images=True)
        return [checkpoint, early, reduceLROnPlat, csv_logger]

    def _fit_siamese(self):

        self.keras_model.compile(optimizer=Adam(self.learning_rate, decay=0.000001),
                                 loss='binary_crossentropy',
                                 metrics=['binary_crossentropy', 'acc'])

        data_generator = PairDataGenerator(self)
        (valid_A_x, valid_B_x), valid_y = data_generator.get_validation_dataset()
        aug_gen = data_generator.get_training_gen()
        callbacks_list = self._get_callbacks()
        loss_hist = [self.keras_model.fit_generator(aug_gen,
                                                    steps_per_epoch=MAX_TRAIN_STEPS,
                                                    epochs=MAX_TRAIN_EPOCHS,
                                                    validation_data=([valid_A_x, valid_B_x], valid_y),
                                                    callbacks=callbacks_list,
                                                    workers=1,
                                                    max_queue_size=1)]
        return loss_hist

    def _fit_triplet_loss(self):
        """
        Start training with triplet loss
        :return: Keras history object
        """
        self.keras_model.compile(optimizer=Adam(self.learning_rate, decay=0.000001),
                                 loss=triplet_loss(margin=self.margin),
                                 metrics=[triplet_acc(margin=0), triplet_acc(margin=self.margin)])

        data_generator = TripletDataGenerator(self)
        aug_gen = data_generator.get_training_gen()

        validation_images, y = data_generator.get_hard_preprocessed_validation_dataset()
        callbacks_list = self._get_callbacks()
        loss_hist = [self.keras_model.fit_generator(aug_gen,
                                                    steps_per_epoch=MAX_TRAIN_STEPS,
                                                    epochs=MAX_TRAIN_EPOCHS,
                                                    validation_data=[validation_images, y],
                                                    callbacks=callbacks_list,
                                                    workers=0,
                                                    max_queue_size=1)]
        return loss_hist

    def get_configuration_parameters_values(self):
        """
        Get all configuration parameters
        :return: Python Tuple with all configuration parameters in a proper order
        """
        return (self.timestamp, self.model_name, self.model.WEIGHT_PATH, self.model.FIT_HISTORY_PATH,
                self.learning_rate, self.load_weights, self.freeze_layers, self.margin,
                self.hard_sampling_batch_size, self.batch_size, self.number_of_validation_imgs,
                self.input_shape)

    def get_configuration_parameters_names(self):
        """
        Get all configuration parameters names
        :return: Python Tuple with all configuration parameters names in a proper order
        """
        return (
            'timestamp', 'model_name', 'weight_path', 'fit_history_path', 'learning_rate', 'load_weights',
            'freeze_layers', 'margin', 'hard_sampling_batch_size', 'batch_size',
            'number_of_validation_imgs', 'input_shape')

    def get_configuration_data(self):
        """
        Get OrderedDict with all parameters in a proper order
        :return: Python OrderedDict with all parameters in a proper order
        """
        keys = self.get_configuration_parameters_names()
        data = self.get_configuration_parameters_values()

        ordered_data = OrderedDict()
        for pos, key in enumerate(keys):
            ordered_data[key] = data[pos]

        return ordered_data

    def _save_configuration_to_csv(self):
        """
        Save configuration to CONFIG_HISTORY_FILE CSV file
        """
        if not os.path.exists(CONFIG_HISTORY_FILE):
            with open(CONFIG_HISTORY_FILE, "w") as f:
                writer = csv.writer(f)
                titles = self.get_configuration_parameters_names()
                writer.writerow(titles)

        with open(CONFIG_HISTORY_FILE, "a") as f:
            writer = csv.writer(f)
            data_row = self.get_configuration_parameters_values()
            writer.writerow(data_row)

    def _save_configuration_to_yml(self):
        """
        Save configuration to "config_" + self.model.timestamp + ".yml" file
        """
        data = self.get_configuration_data()
        timestamp = self.model.timestamp
        with open(os.path.join(CHECKPOINTS_DIR, timestamp, 'config_{}.yml'.format(timestamp)), 'w') as outfile:
            yaml.dump(dict(data), outfile, default_flow_style=False)

    def train(self):
        """
        Start training
        :return: Keras history object
        """

        if self.siamese_model:
            loss_hist = self._fit_triplet_loss()
        else:
            foss_hist = self._fit_siemese()

        print('Saving configuration to yml and csv files')
        self._save_configuration_to_yml()
        self._save_configuration_to_csv()

        return loss_hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-mn', '--model_name', default=DEFAULT_MODEL_NAME, help="Specify name of the model to use")
    ap.add_argument('-lr', '--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
    ap.add_argument('-lw', '--load_weights', action='store_true', default=DEFAULT_LOAD_WEIGHTS,
                    help='Specify if try load weights before training. Needs to have --timestamp parameter specified.')
    ap.add_argument('-tc', '--training_config', type=str, default=DEFAULT_TRAINING_CONFIG,
                    help='Config file to load, by default configuration is taken from config.py')
    ap.add_argument('-fr', '--freeze_layers', type=int, default=DEFAULT_FREEZE_LAYERS,
                    help="Specify number of layers that should be non-trainable")
    ap.add_argument('-bs', '--batch_size', type=int, default=BATCH_SIZE)
    ap.add_argument('-vi', '--number_of_validation_imgs', type=int, default=VALID_IMG_NUM,
                    help="Specify size of validation dataset. By default it uses 5000 samples.")
    ap.add_argument('-is', '--input_shape', nargs='+', type=int, default=INPUT_SHAPE,
                    help="Specify images input shape to Neural Network. i.e.: -is 224 224 3")
    ap.add_argument('-mr', '--margin', type=float, default=DEFAULT_MARGIN,
                    help='Value of margin if triplet loss used')
    ap.add_argument('-hb', '--hard_sampling_batch_size', type=int, default=DEFAULT_HARD_SAMPLING_BATCH,
                    help='If triplet loss and hard sampling is used then specify among how many samples'
                         'we should search for a hard one during training step')
    ap.add_argument('-ts', '--timestamp', type=str, default=DEFAULT_TIMESTAMP,
                    help='Specify if you want to use one of old timestamp directories, i.e. for evaluation purposes.')

    kwargs = vars(ap.parse_args())

    if kwargs['training_config'] is not None:
        with open(kwargs['training_config'], 'r') as cfg_file:
            kwargs = yaml.load(cfg_file)

    train = TrainingConfigurator(kwargs)
    loss_history = train.train()

    gc.collect()
