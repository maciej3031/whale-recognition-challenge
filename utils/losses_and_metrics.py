import keras.backend as K
import tensorflow as tf

from config import DEFAULT_MARGIN


def triplet_loss(margin=DEFAULT_MARGIN):
    def _triplet_loss(y_true, y_pred):
        """
        Implementation of the triplet loss function
        :param y_true: true labels, required by Keras, not used in calculations.
        :param y_pred: np.array with shape (BATCH_SIZE, 3, VECTOR_LENGTH):
                       anchor -- the embedding for the anchor data
                       positive -- the embedding for the positive data (similar to anchor)
                       negative -- the embedding for the negative data (different from anchor)
        :return: loss value
        """
#         y_pred = tf.Print(y_pred, [y_pred], message="This is y_pred: ")
        anchor = y_pred[:, 0, :]
        positive = y_pred[:, 1, :]
        negative = y_pred[:, 2, :]

        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor - positive), axis=1)

        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor - negative), axis=1)

        # compute loss
        basic_loss = pos_dist - neg_dist + margin
        loss = K.maximum(basic_loss, 0.0)

        return loss

    return _triplet_loss


def triplet_acc(margin=0):
    def _triplet_acc(y_true, y_pred):
        """
        Implementation of the triplet accuracy.
        Number of samples where positive distance < negative distance divided by all samples.
        :param y_true: true labels, required by Keras, not used in calculations.
        :param y_pred: np.array with shape (BATCH_SIZE, 3, VECTOR_LENGTH):
                       anchor -- the embedding for the anchor data
                       positive -- the embedding for the positive data (similar to anchor)
                       negative -- the embedding for the negative data (different from anchor)
        :return: accuracy value
        """
        anchor = y_pred[:, 0, :]
        positive = y_pred[:, 1, :]
        negative = y_pred[:, 2, :]

        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor - positive), axis=1)

        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor - negative), axis=1)

        diff = neg_dist - pos_dist - margin

        diff_bool = diff > 0
        count_good = K.sum(tf.cast(diff_bool, tf.int32))
        sum_all = tf.size(pos_dist)

        acc = count_good / sum_all

        return acc

    return _triplet_acc
