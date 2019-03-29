import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import scipy
from sklearn import metrics
from scipy import stats

import json


class perf_biometric_proba():
    def __init__(self, pred_neg=None, pred_pos=None):
        """Initialise class"""
        self.pred_neg = pred_neg
        self.pred_pos = pred_pos
        self.labels = ['neg', 'pos']
        return

    def set_labels(self, labels):
        self.labels = labels

    def save_scores(self, filename_json):
        data = {'labels': self.labels,
                'pred_neg': self.pred_neg.tolist(),
                'pred_pos': self.pred_pos.tolist()}
        with open(filename_json, 'w') as outfile:
            json.dump(data, outfile)

    def load_scores(self, filename_json):
        with open(filename_json) as f:
            data = json.load(f)
        self.labels = data['labels']
        self.pred_neg = np.array(data['pred_neg'])
        self.pred_pos = np.array(data['pred_pos'])
        return

    def get_param(self):
        return self.thrd_param

    def set_param(self, thrd_param):
        self.thrd_param = thrd_param
        return

    def fit(self):
        if min(self.pred_neg) < 0 or max(self.pred_neg) > 1 or min(self.pred_pos) < 0 or max(self.pred_pos) > 1:
            raise Exception('Scores are not probability')

        self.thrd_param = perf_calibrate_scores_get_parameter(self.pred_neg, self.pred_pos)
        self.far, self.frr, self.x = perf_get_far_frr(self.pred_neg, self.pred_pos)
        return

    def transform(self, scores):
        return perf_calibrate_scores(scores, self.thrd_param)

    def plot_all(self):
        f, axarr = plt.subplots(1, 3)
        perf_plot_DET_ax(self.far, self.frr, axarr[0])
        axarr[0].axis([0.001, 1, 0.001, 1])
        perf_plot_far_frr(self.far, self.frr, self.x, axarr[1])
        axarr[1].grid(True)
        perf_plot_hist(self.pred_neg, self.pred_pos, ax=axarr[2], labels=self.labels)
        return

    def plot_all_calibrated(self):
        return perf_plot_all(self.transform(self.pred_neg),
                             self.transform(self.pred_pos), labels=self.labels)

    def get_f_ratio(self):
        f_ratio = perf_f_ratio(self.transform(self.pred_neg), self.transform(self.pred_pos))
        return f_ratio

    def plot_DET(self):
        return perf_plot_DET(self.far, self.frr)

    def plot_ROC(self):
        return perf_plot_ROC(self.far, self.frr)

    def plot_hist(self, ax=None):
        return perf_plot_hist(self.pred_neg, self.pred_pos, nbins=None, ax=ax, labels=self.labels)

    def plot_far_frr(self):
        return perf_plot_far_frr(self.far, self.frr, self.x)

    def get_metrics(self):
        eer, eer_thrd, _ = perf_eer(self.far, self.frr, self.x)
        min_hter, min_hter_thrd, _ = perf_hter(self.far, self.frr, self.x)

        return {'eer': eer, 'eer_thrd': eer_thrd, 'min_hter': min_hter, 'min_hter_thrd': min_hter_thrd,
                'eer_theory': perf_eer_theory(self.get_f_ratio())}

    def print_metrics(self):
        dict_ = self.get_metrics()
        print('EER theory {:.2f}%'.format(dict_['eer_theory'] * 100))
        print('EER {:.2f}%'.format(dict_['eer'] * 100))
        print('min HTER {:.2f}%'.format(dict_['min_hter'] * 100))
        print('threshold at EER {:.4f}'.format(dict_['eer_thrd']))
        print('threshold at min HTER {:.4f}'.format(dict_['min_hter_thrd']))


def perf_f_ratio(score0, score1):
    return (np.mean(score1) - np.mean(score0)) / (np.std(score0) + np.std(score1))


def perf_eer_theory(f_ratio):
    return 0.5 - 0.5 * scipy.special.erf(f_ratio / np.sqrt(2))


# all performance-related functions start with perf
def perf_plot_DET(fps, fns):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    far = fpr, frr = fnr
    """
    # https://jeremykarnowski.wordpress.com/2015/08/07/detection-error-tradeoff-det-curves/
    axis_min = min(fps[0], fns[-1])
    fig, ax = plt.subplots()
    plt.plot(fps, fns)
    plt.yscale('log')
    plt.xscale('log')
    ticks_to_use = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.axis([0.001, 50, 0.001, 50])
    plt.xlabel('FAR')
    plt.ylabel('FRR')
    plt.grid()


def perf_plot_DET_ax(fps, fns, ax, color='blue', label='DET'):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    far = fpr, frr = fnr
    """
    # https://jeremykarnowski.wordpress.com/2015/08/07/detection-error-tradeoff-det-curves/
    axis_min = min(fps[0], fns[-1])
    # fig,ax = plt.subplots()
    ax.plot(fps, fns, color=color, label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ticks_to_use = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    ax.axis([0.001, 50, 0.001, 50])
    ax.grid()


def perf_print_metrics(scores_imp, scores_gen, return_value=False):
    far, frr, x = perf_get_far_frr(scores_imp, scores_gen)
    min_HTER, thrd_min_HTER, _ = perf_hter(far, frr, x)
    print('min_HTER : {:.2f}%'.format(min_HTER * 100))
    print('threshold at min_HTER : {:.4f}'.format(thrd_min_HTER))
    eer, thrd_eer, _ = perf_eer(far, frr, x)
    print('EER : {:.2f}%'.format(eer * 100))
    print('threshold at EER : {:.4f}'.format(thrd_eer))
    if return_value:
        return min_HTER, eer, thrd_min_HTER, thrd_eer
    else:
        return


def perf_plot_density(scores_imp, scores_gen, thresholds, ax=None, labels=['neg', 'pos']):
    if ax is None:
        _, ax = plt.subplots()
    lw = 2
    kde_gen = stats.gaussian_kde(scores_gen)
    kde_imp = stats.gaussian_kde(scores_imp)
    x = np.linspace(thresholds.min(), thresholds.max(), 100)
    ax.plot(x, kde_gen(x), lw=lw, label=labels[1])
    ax.plot(x, kde_imp(x), lw=lw, label=labels[0])


def perf_plot_hist(scores_imp, scores_gen, ax=None, nbins=None, labels=['neg', 'pos']):
    if nbins is None:
        if len(scores_gen) > len(scores_imp):
            ratio_ = len(scores_gen) / len(scores_imp)
            nbins = [20, int(np.floor(ratio_ * 20))]
        else:
            ratio_ = len(scores_imp) / len(scores_gen)
            nbins = [int(np.floor(ratio_ * 20)), 20]

    if ax is None:
        _, ax = plt.subplots()
    lw = 2
    # bins = np.linspace(thresholds.min(), thresholds.max(), nbins)
    # ax.hist(scores_imp, bins, lw=lw, alpha=0.5, label='imp', color='red')
    ax.hist(scores_imp, bins=nbins[0], density=True, lw=lw, alpha=0.5, label=labels[0], color='red')
    # bins = np.linspace(thresholds.min(), thresholds.max(), int(len(scores_gen)/len(scores_imp) * nbins))
    ax.hist(scores_gen, bins=nbins[1], density=True, lw=lw, alpha=0.5, label=labels[1], color='blue')
    ax.legend(loc="upper right")


def perf_plot_far_frr(far, frr, thresholds, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    lw = 2
    ax.plot(thresholds, far, color='red', lw=lw, label='FAR')
    ax.plot(thresholds, frr, color='blue', lw=lw, label='FRR')
    ax.legend(loc="upper right")
    ax.autoscale(enable=True, axis='x', tight=True)


def perf_plot_ROC(far, frr):
    plt.figure();
    lw = 2
    plt.plot(far, frr, color='red', lw=lw)
    plt.grid()
    plt.xlabel('FAR')
    plt.ylabel('FRR')


def perf_plot_all(scores_imp, scores_gen, use_density=False, labels=None):
    far, frr, thresholds = perf_get_far_frr(scores_imp, scores_gen)
    f, axarr = plt.subplots(1, 3)
    perf_plot_DET_ax(far, frr, axarr[0])
    axarr[0].axis([0.001, 1, 0.001, 1])
    perf_plot_far_frr(far, frr, thresholds, axarr[1])
    axarr[1].grid(True)

    if use_density:
        perf_plot_density(scores_imp, scores_gen, thresholds, axarr[2], labels=labels)
    else:
        perf_plot_hist(scores_imp, scores_gen, axarr[2], labels=labels)
    return axarr


def perf_get_scores(D, label_keys):
    """
    :param D: A square distance matrix
    :param label_keys: a list of random key representation of the identities. len(label_keys)
    :return: scores_imp, scores_gen
    """

    # compute mask for the genuine scores from the key
    mask = metrics.pairwise.pairwise_distances(label_keys)
    mask_gen = np.asarray(mask < 0.0001) * 1.0
    mask_gen = np.triu(mask_gen) - np.identity(mask_gen.shape[0])
    # plt.imshow(mask_gen)
    # plt.colorbar()
    # plt.show()

    # compute mask for impostor
    mask_imp = np.ones(mask_gen.shape) - mask_gen - np.identity(mask_gen.shape[0])
    mask_imp = np.triu(mask_imp)
    # plt.imshow(mask_imp); plt.colorbar(); plt.show()

    # just checking
    # plt.imshow(mask_imp+mask_gen)

    # scores are symmetrical and so we need only half of them
    # plt.imshow(np.multiply(D, mask_gen));
    # plt.colorbar()

    indice = np.nonzero(mask_gen)
    scores_gen = D[indice[0], indice[1]]
    indice = np.nonzero(mask_imp)
    scores_imp = D[indice[0], indice[1]]
    return scores_imp, scores_gen


def perf_logit(scores):
    return np.log(scores + np.sqrt(np.finfo(float).eps)) - np.log(1 - scores + np.sqrt(np.finfo(float).eps))


def perf_sigmoid(scores, shift=0, scale=1):
    return 1 / (1 + np.exp(- (scores * scale + shift)))


def perf_calibrate_scores_get_parameter(preds_fake, preds_real):
    # transform scores to have a threshold with EER exactly = 0.5
    pred_fake = perf_logit(preds_fake)
    pred_real = perf_logit(preds_real)

    far, frr, x = perf_get_far_frr(pred_fake, pred_real)
    _, thrd_param, _ = perf_eer(far, frr, x)
    return thrd_param


def perf_calibrate_scores(preds_fake, thrd=None):
    # transform scores to have a threshold with EER exactly = 0.5
    pred_fake = perf_logit(preds_fake)
    pred_fake_ = perf_sigmoid(pred_fake, -thrd)
    return pred_fake_


def perf_calibrate_scores_two_sets(preds_fake, preds_real, thrd=None):
    # transform scores to have a threshold with EER exactly = 0.5

    pred_fake = perf_logit(preds_fake)
    pred_real = perf_logit(preds_real)

    if thrd is None:
        far, frr, x = perf_get_far_frr(pred_fake, pred_real)
        eer, thrd, _ = perf_eer(far, frr, x)

    pred_fake_ = perf_sigmoid(pred_fake, -thrd)
    pred_real_ = perf_sigmoid(pred_real, -thrd)
    return pred_fake_, pred_real_


def perf_get_far_frr(scores_imp, scores_gen, reverse_sign=False):
    if reverse_sign:
        scores_ = -1 * np.concatenate((scores_gen, scores_imp), axis=0)
    else:
        scores_ = np.concatenate((scores_gen, scores_imp), axis=0)

    label_ = np.concatenate((np.ones(scores_gen.shape), np.zeros(scores_imp.shape)), axis=0)
    fpr, tpr, thresholds = metrics.roc_curve(label_, scores_, pos_label=1)
    far = fpr
    frr = 1 - tpr

    if reverse_sign:
        thresholds = thresholds[1:]
        far = far[1:]
        frr = frr[1:]
    return far, frr, thresholds


# EER
def perf_eer(far, frr, thresholds, reverse_sign=False):
    index_eer = np.argmin(abs(far - frr), axis=0)
    min_eer = (far[index_eer] + frr[index_eer]) / 2
    thrd_at_eer = thresholds[index_eer]
    if reverse_sign:
        return min_eer, - thrd_at_eer, index_eer
    else:
        return min_eer, thrd_at_eer, index_eer


# HTER
def perf_hter(far, frr, thresholds, reverse_sign=False):
    index_hter = np.argmin((far + frr) / 2, axis=0)
    min_hter = (far[index_hter] + frr[index_hter]) / 2
    thrd_at_hter = thresholds[index_hter]
    if reverse_sign:
        return min_hter, - thrd_at_hter, index_hter
    else:
        return min_hter, thrd_at_hter, index_hter
