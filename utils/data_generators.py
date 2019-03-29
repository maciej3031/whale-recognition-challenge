import gc
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd

from config import DATA_DIR, SEQ, K, P

gc.enable()  # memory is tight

from utils.utils import preprocess_input, load_image, prepare_tensor_from_image


class DataGenerator:
    def __init__(self, train_conf):
        self.train_conf = train_conf

        # random value need for generating labels for triplet loss, which are not used but keras requires it...
        self.length = 1234

    def preprocess_input(self, x):
        """
        Preprocess input in accordance to Keras preprocessing standards
        if self.train_conf.normalization_type == DEFAULT_NORMALIZATION_TYPE
        :param x: Numpy array with values between 0 and 255
        :return: Numpy array with preprocessed values
        """
        return preprocess_input(x, self.train_conf.model_name)

    def load_image(self, image_name, mode):
        """
        Load image and convert to Numpy array with values between 0 and 1
        :param image_name: String with image name for ALL_DIR directory
        :param mode: String "train" or "test", depends on which folder take images from
        :return: Numpy array with values between 0 and 1 with shape self.train_conf.input_shape
        """
        return load_image(image_name, self.train_conf.input_shape, mode)

    def prepare_tensor_from_image(self, img):
        """
        Expand first dimension of input Numpy array:
        self.train_conf.input_shape -> (1, *self.train_conf.input_shape)
        And preprocces its values using Keras preprocessing
        :param img: Numpy array with values between 0 and 1 with shape self.train_conf.input_shape
        :return: Numpy array with values between 0 and 1 with shape (1, *self.train_conf.input_shape)
        """
        return prepare_tensor_from_image(img, self.train_conf.model_name)

    def get_all_dateset_by_mode(self, mode):
        """
        Get dataset with all whales as Pandas Dataframe by mode (train or validation).
        :param mode: String 'train' or 'validation'
        :return: Pandas Dataframe
        """
        if mode == "train":
            csv_file = "train_all.csv"
        elif mode == "validation":
            csv_file = "validation_all.csv"
        else:
            raise Exception("No such mode!")

        return csv_file

    def get_random_negative_img(self, whale_id, mode):
        """
        Get random negative image to some other image given by whale ID
        :param whale_id: String with full image name
        :param mode: String "train" or "test", depends on which folder take images from
        :return: Numpy array with values between 0 and 1 with shape self.train_conf.input_shape
        """
        csv_file = self.get_all_dateset_by_mode(mode)
        all_images = pd.read_csv(os.path.join(DATA_DIR, csv_file))
        all_negative_images = all_images.loc[all_images["Id"] != whale_id]
        img_N_name = all_negative_images.sample()["Image"].values[0]
        img_N = self.load_image(img_N_name, mode)

        return img_N

    def get_random_negative_batch(self, whale_id, mode):
        """
        Get random negative batch of images names to some image given by whale ID
        :param whale_id: String with full image name
        :param mode: String "train" or "test", depends on which folder take images from
        :return: Pandas Dataframe with images names with shape: (self.train_conf.hard_sampling_batch_size, 1)
        """
        csv_file = self.get_all_dateset_by_mode(mode)
        all_images = pd.read_csv(os.path.join(DATA_DIR, csv_file))
        all_negative_images = all_images.loc[all_images["Id"] != whale_id]
        negative_samples = all_negative_images.sample(
            self.train_conf.hard_sampling_batch_size
        )

        return negative_samples

    def is_hard_or_semi_hard_sample(self, A_embedding, P_embedding, N_embedding):
        """
        Check if negative sample is within margin.
        :param A_embedding: Numpy array, with shape (1,N)
        :param P_embedding: Numpy array, with shape (1,N)
        :param N_embedding: Numpy array, with shape (1,N)
        :return: Bool
        """
        pos_dist = np.sum(np.square(A_embedding - P_embedding), axis=1)
        neg_dist = np.sum(np.square(A_embedding - N_embedding), axis=1)

        diff = pos_dist - neg_dist + self.train_conf.margin

        return diff >= 0

    def is_hard_sample(self, A_embedding, P_embedding, N_embedding):
        """
        Check if negative sample is within margin.
        :param A_embedding: Numpy array, with shape (1,N)
        :param P_embedding: Numpy array, with shape (1,N)
        :param N_embedding: Numpy array, with shape (1,N)
        :return: Bool
        """
        pos_dist = np.sum(np.square(A_embedding - P_embedding), axis=1)
        neg_dist = np.sum(np.square(A_embedding - N_embedding), axis=1)

        diff = pos_dist - neg_dist

        return diff >= 0

    def is_semi_hard_sample(self, A_embedding, P_embedding, N_embedding):
        """
        Check if negative sample is within margin.
        :param A_embedding: Numpy array, with shape (1,N)
        :param P_embedding: Numpy array, with shape (1,N)
        :param N_embedding: Numpy array, with shape (1,N)
        :return: Bool
        """
        pos_dist = np.sum(np.square(A_embedding - P_embedding), axis=1)
        neg_dist = np.sum(np.square(A_embedding - N_embedding), axis=1)

        semi_hard_diff = pos_dist - neg_dist + 0.2  # Not margin but only its part to have harder samples
        hard_diff = pos_dist - neg_dist

        return semi_hard_diff >= 0 and hard_diff <= 0


class PairDataGenerator(DataGenerator):
    def make_image_gen(self):
        df_train_all = pd.read_csv(os.path.join(DATA_DIR, 'train_pairs.csv'))

        positive = df_train_all.loc[df_train_all['the_same'] is True]
        negative = df_train_all.loc[df_train_all['the_same'] is False]

        while True:
            positive_samples = positive.sample(self.train_conf.batch_size // 2)
            negative_samples = negative.sample(self.train_conf.batch_size // 2)

            samples = pd.concat([positive_samples, negative_samples]).values
            np.random.shuffle(samples)

            A_images, B_images, labels = [], [], []
            for image_A_name, image_B_name, label in samples:
                img_A = self.load_image(image_A_name, mode='train')
                img_B = self.load_image(image_B_name, mode='train')

                A_images.append(img_A)
                B_images.append(img_B)
                labels.append([label])

            yield [np.stack(A_images, 0), np.stack(B_images, 0)], np.stack(labels)
            gc.collect()

    def create_aug_gen(self, in_gen):
        for (imgA, imgB), y in in_gen:
            g_imgA = SEQ.augment_images(imgA)
            g_imgB = SEQ.augment_images(imgB)

            yield [
                      g_imgA.astype(np.uint8),
                      g_imgB.astype(np.uint8),
                  ], y

    def preprocess_input_gen(self, in_gen):
        for [imgA, imgB], label in in_gen:
            imgA = self.preprocess_input(imgA)
            imgB = self.preprocess_input(imgB)
            yield [imgA, imgB], label

    def get_training_gen(self):
        return self.preprocess_input_gen(self.create_aug_gen(self.make_image_gen()))

    def get_validation_dataset(self):
        df_validation_all = pd.read_csv(os.path.join(DATA_DIR, 'validation_pairs.csv'))

        positive = df_validation_all.loc[df_validation_all['the_same'] is True]
        negative = df_validation_all.loc[df_validation_all['the_same'] is False]

        positive_samples = positive.sample(self.train_conf.number_of_validation_imgs // 2)
        negative_samples = negative.sample(self.train_conf.number_of_validation_imgs // 2)

        samples = pd.concat([positive_samples, negative_samples]).values
        np.random.shuffle(samples)

        A_images, B_images, labels = [], [], []
        for image_A_name, image_B_name, label in samples:
            img_A = self.load_image(image_A_name, mode='train')
            img_B = self.load_image(image_B_name, mode='train')

            A_images.append(img_A)
            B_images.append(img_B)
            labels.append([label])

        return [self.preprocess_input(np.stack(A_images, 0)),
                self.preprocess_input(np.stack(B_images, 0))], np.stack(labels)


class TripletDataGenerator(DataGenerator):
    def _get_hard_negative_img(self, whale_id, img_A, img_P, mode):
        """
        Get image that is consider to be "semi-hard" to distinguish for NN from img_A.
        In oder words "semi-hard" means:
        distance between embeddings of img_A and img_P + margin > distance between embeddings of im_A and negative image
        :param Id: String, Id of anchor image
        :param img_A: Numpy array, anchor image
        :param img_P: Numpy array, positive image
        :return: Numpy array, found, semi-hard, negative image
        """

        batch_of_negative_images_names = self.get_random_negative_batch(
            whale_id, mode=mode
        )
        for image_N_name, _ in batch_of_negative_images_names.values:
            tensor_A = self.prepare_tensor_from_image(img_A)
            tensor_P = self.prepare_tensor_from_image(img_P)

            img_N = self.load_image(image_N_name, mode)
            tensor_N = self.prepare_tensor_from_image(img_N)

            pred = self.train_conf.keras_model.predict([tensor_A, tensor_P, tensor_N])

            anchor_embedding = pred[:, 0, :]
            positive_embedding = pred[:, 1, :]
            negative_embedding = pred[:, 2, :]

            if self.is_semi_hard_sample(
                    anchor_embedding, positive_embedding, negative_embedding
            ):
                return img_N

        return self.get_random_negative_img(whale_id, mode=mode)

    def _get_distances_between_A_and_candidates(self, im_A, im_candidates):
        tmp_im_A = deepcopy(im_A)
        a_tensor = self.preprocess_input(np.stack([tmp_im_A for i in range(len(im_candidates))], 0))
        p_tensor = self.preprocess_input(np.stack(im_candidates, 0))

        A_embeddings = self.train_conf.encoder.predict(a_tensor)
        other_embeddings = self.train_conf.encoder.predict(p_tensor)
        dist = np.sum(np.square(A_embeddings - other_embeddings), axis=1)
        return dist

    def make_image_gen_hard_batch(self):
        """
        Create generator of triplets and random labels (not used in Triplet Loss training but required by Keras)
        Triplets consist of Anchor, Positive image and Negative image. Triplets are created as described in:
        https://arxiv.org/pdf/1703.07737.pdf (formula (5), page 3)
        :return: Python generator object that generates: ([Numpy array, Numpy array, Numpy array], Numpy array)
        """
        df_train_all = pd.read_csv(os.path.join(DATA_DIR, 'train_all.csv'))

        df_train_no_new_whales = df_train_all[df_train_all['Id'] != 'new_whale']
        grouped = df_train_no_new_whales.groupby('Id')
        df_train_two_or_more = grouped.filter(lambda x: len(x) >= 2)

        classes = list(set(list(df_train_two_or_more['Id'].unique())))

        while True:
            chosen_classes = random.sample(classes, P)
            batch_samples = []
            for chosen_Id in chosen_classes:
                maches = df_train_two_or_more[df_train_two_or_more['Id'] == chosen_Id].values
                np.random.shuffle(maches)
                chosen_matches = maches[:K]
                batch_samples.append(chosen_matches)
            batch_samples = np.concatenate(batch_samples, axis=0)
            np.random.shuffle(batch_samples)
            batch_samples = pd.DataFrame(batch_samples, columns=["Image", "Id"])

            A_images, P_images, N_images = [], [], []
            for im_A_name, im_A_Id in batch_samples.values:
                im_A = self.load_image(im_A_name, mode='train')

                # Get furthest positive sample within batch
                im_P_name_candidates = batch_samples[batch_samples['Id'] == im_A_Id][
                    batch_samples['Image'] != im_A_name]
                im_P_candidates = []
                for im_P_name, _ in im_P_name_candidates.values:
                    im_P = self.load_image(im_P_name, mode='train')
                    im_P_candidates.append(im_P)

                pos_distances = self._get_distances_between_A_and_candidates(im_A, im_P_candidates)

                chosen_positive_pair_index = np.argmax(pos_distances)
                im_P = im_P_candidates[chosen_positive_pair_index]
                print("Img P collected: ")
                # get closest negative sample within batch
                im_N_name_candidates = batch_samples[batch_samples['Id'] != im_A_Id]
                im_N_candidates = []
                for im_N_name, _ in im_N_name_candidates.values:
                    im_N = self.load_image(im_N_name, mode='train')
                    im_N_candidates.append(im_N)

                neg_distances = self._get_distances_between_A_and_candidates(im_A, im_N_candidates)

                chosen_negative_pair_index = np.argmin(neg_distances)
                im_N = im_N_candidates[chosen_negative_pair_index]
                print("Img N collected: ")

                A_images.append(im_A)
                P_images.append(im_P)
                N_images.append(im_N)

                if len(A_images) >= self.train_conf.batch_size:
                    yield [np.stack(A_images, 0), np.stack(P_images, 0), np.stack(N_images, 0)], \
                          np.random.rand(len(A_images), 3, self.length)
                    A_images, P_images, N_images = [], [], []

            gc.collect()

    def make_image_gen(self):
        """
        Create generator of triplets and random labels (not used in Triplet Loss training but required by Keras)
        Triplets consist of Anchor, Positive image and Negative image. Triplets are created in a following steps:
        1. Take random positive pair
        2. Take batch of random negative images
        3. In a taken batch, find negative image that is considered to be "semi-hard" for NN.
        If not found, take random negative image
        More details in self._get_hard_negative_img()
        :param csv_file: String, name of the CSV file with data
        :return: Python generator object that generates: ([Numpy array, Numpy array, Numpy array], Numpy array)
        """
        positive_pairs = pd.read_csv(os.path.join(DATA_DIR, "train_positive_pairs.csv"))
        while True:
            positive_samples = positive_pairs.sample(self.train_conf.batch_size).values
            A_images, P_images, N_images = [], [], []

            for image_A_name, image_P_name, Id in positive_samples:
                tmp_img_A = self.load_image(image_A_name, mode='train')
                tmp_img_P = self.load_image(image_P_name, mode='train')

                img_A, img_P = (tmp_img_A, tmp_img_P) if np.random.rand() > 0.5 else (tmp_img_P, tmp_img_A)

                img_N = self._get_hard_negative_img(Id, img_A, img_P, mode='train')

                A_images.append(img_A)
                P_images.append(img_P)
                N_images.append(img_N)
            yield [
                      np.stack(A_images, 0),
                      np.stack(P_images, 0),
                      np.stack(N_images, 0),
                  ], np.random.rand(len(A_images), self.length)
            #             print(self.train_conf.encoder.get_layer('block14_sepconv2_bn').get_weights()[2], end='\n')
            #             print(self.train_conf.encoder.get_layer('batch_normalization_4').get_weights()[2], end='\n')
            gc.collect()

    def create_aug_gen(self, in_gen):
        """
        Augment data using Keras Data Augmentation
        :param in_gen: Python generator object that generates: ([Numpy array, Numpy array, Numpy array], Numpy array)
        :return: Python generator object that generates: ([Numpy array, Numpy array, Numpy array], Numpy array)
        """

        for (imgA, imgP, imgN), y in in_gen:
            g_imgA = SEQ.augment_images(imgA)
            g_imgP = SEQ.augment_images(imgP)
            g_imgN = SEQ.augment_images(imgN)

            yield [
                      g_imgA.astype(np.uint8),
                      g_imgP.astype(np.uint8),
                      g_imgN.astype(np.uint8),
                  ], y
            gc.collect()

    def preprocess_input_gen(self, in_gen):
        """
        Preprocces data from given generator
        :param in_gen: Python generator object that generates: ([Numpy array, Numpy array, Numpy array], Numpy array)
        :return: Python generator object that generates: ([Numpy array, Numpy array, Numpy array], Numpy array)
        """
        for (imgA, imgP, imgN), y in in_gen:
            imgA = self.preprocess_input(imgA)
            imgP = self.preprocess_input(imgP)
            imgN = self.preprocess_input(imgN)

            yield [imgA, imgP, imgN], y

    def get_training_gen(self):
        """
        Create generator for NN training
        :param csv_file: String, name of the CSV file with data
        :return: Python generator object
        """
        if self.train_conf.hard_batch_approach:
            return self.preprocess_input_gen(self.create_aug_gen(self.make_image_gen_hard_batch()))
        else:
            return self.preprocess_input_gen(self.create_aug_gen(self.make_image_gen()))

    def _get_all_triplets(self, positives, all_whales_list):
        """
        Create all possible triplets given list of positives pairs and all whales list
        :param positives: Numpy array with shape (n,3) or List of 3-element tuples
        (Positive pairs of images names and whale Id)
        :param all_whales_list: Numpy array with shape (n,2) or List of 2-element tuples,
        that are all possible whale images and IDs
        :return: List o 3-element Tuples
        """
        triplets = []
        for a in positives:
            A_name, P_name, A_id = a
            for N_name, N_id in all_whales_list:
                if A_id != N_id:
                    triplets.append((A_name, P_name, N_name))
        return triplets

    def get_all_validation_samples_names(self, shuffle=True):
        """
        Prepare all possible triplets
        :param shuffle: Boolean
        :return: Python List of 3-element Tuples
        """
        positive_pairs = pd.read_csv(
            os.path.join(DATA_DIR, "validation_positive_pairs.csv")
        )
        all_validation_images_names = pd.read_csv(
            os.path.join(DATA_DIR, "validation_all.csv")
        )

        all_triplets = self._get_all_triplets(
            positive_pairs.values, all_validation_images_names.values
        )

        if shuffle:
            random.shuffle(all_triplets)

        return all_triplets

    def get_hard_preprocessed_validation_dataset(self):
        """
        Create validation dataset with only semi-hard samples
        :return: [Numpy array, Numpy array, Numpy array], Numpy array
        """
        print("Validation dataset creation...")
        triplets = self.get_all_validation_samples_names()
        A_images, P_images, N_images = [], [], []
        for A, P, N in triplets:
            img_A = self.load_image(A, mode="train")
            img_P = self.load_image(P, mode="train")
            img_N = self.load_image(N, mode="train")

            tensor_A = self.prepare_tensor_from_image(img_A)
            tensor_P = self.prepare_tensor_from_image(img_P)
            tensor_N = self.prepare_tensor_from_image(img_N)

            A_embedding = self.train_conf.encoder.predict(tensor_A)
            P_embedding = self.train_conf.encoder.predict(tensor_P)
            N_embedding = self.train_conf.encoder.predict(tensor_N)

            if self.is_hard_or_semi_hard_sample(A_embedding, P_embedding, N_embedding):
                A_images.append(img_A)
                P_images.append(img_P)
                N_images.append(img_N)

            if len(A_images) >= self.train_conf.number_of_validation_imgs:
                break

        print("Validation dataset consists of {} samples".format(len(A_images)))
        gc.collect()
        return (
            [
                self.preprocess_input(np.stack(A_images, 0)),
                self.preprocess_input(np.stack(P_images, 0)),
                self.preprocess_input(np.stack(N_images, 0)),
            ],
            np.random.rand(len(A_images), 3, self.length),
        )
