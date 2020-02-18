# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
import math
import matplotlib
import numpy as np
from scipy import interp
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

from upclass.uniprot.input.utils import load_mlb

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def plot_curve(y_real, y_pred, n_classes, p_name=None):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    lw = 1  # line weight
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(y_real[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], threshold['micro'] = roc_curve(y_real.ravel(), y_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Find the best cut-off of the ROC curve
    dist_roc = 1
    for i in range(len(tpr['micro'])):
        dist_i = math.sqrt(math.pow(1 - tpr['micro'][i], 2) + math.pow(fpr['micro'][i], 2))
        if dist_roc > dist_i:
            dist_roc = dist_i
            threshold['best'] = threshold['micro'][i]
    print('threshold (dist %.2f): %.2f' % (dist_roc, threshold['best']))

    # Plot all ROC curves
    fig = plt.figure()
    plt.plot(fpr['micro'], tpr['micro'],
             label='micro-average (area = {0:0.2f})'
                   ''.format(roc_auc['micro']),
             linestyle='-', color='dodgerblue', linewidth=2, alpha=.9)
    #

    plt.plot(fpr['macro'], tpr['macro'],
             label='macro-average (area = {0:0.2f})'
                   ''.format(roc_auc['macro']),
             linestyle='-', color='orangered', linewidth=2, alpha=.9)
    # linestyle = ':',

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    x = np.linspace(0.0, 1.0, n_classes)

    rgb = matplotlib.cm.get_cmap(plt.get_cmap('tab20'))(x)[np.newaxis, :, :3][0]

    mlb = load_mlb()

    for i, color in zip(range(n_classes), rgb):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=1, linestyle='--',
                 label='{0} (area = {1:0.2f})'
                       ''.format(mlb.classes_[i], roc_auc[i]), alpha=.8)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC for UniProt classes')
    plt.legend(loc='lower right')
    # plt.show()
    if p_name is None:
        p_name = 'roc_fig'
    fig.savefig(p_name + '.png', bbox_inches='tight')

    return threshold['best']


def compute_sens_threshold(y_real, y_pred, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    lw = 1  # line weight
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(y_real[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], threshold['micro'] = roc_curve(y_real.ravel(), y_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Find the best cut-off of the ROC curve
    import math
    dist_roc = 1
    for i in range(len(tpr['micro'])):
        dist_i = math.sqrt(math.pow(1 - tpr['micro'][i], 2) + math.pow(fpr['micro'][i], 2))
        if dist_roc > dist_i:
            dist_roc = dist_i
            threshold['best'] = threshold['micro'][i]
    print('threshold (dist %.2f): %.2f' % (dist_roc, threshold['best']))

    return fpr, tpr, roc_auc, threshold['best']


def plot_prec_rec_curve(y_real, y_pred, n_classes, p_name=None):
    prec, rec, avg_prec, threshold = compute_threshold(y_real, y_pred, n_classes)

    plot_metric(prec, rec, avg_prec, p_name=p_name)

    return threshold['best']


def compute_metrics(y_real, y_pred, n_classes):
    prec = dict()
    rec = dict()
    threshold = dict()
    avg_prec = dict()
    for i in range(n_classes):
        prec[i], rec[i], threshold[i] = precision_recall_curve(y_real[:, i], y_pred[:, i])
        avg_prec[i] = average_precision_score(y_real[:, i], y_pred[:, i])

    # Compute micro-average precision recall curve and average precision
    prec['micro'], rec['micro'], threshold['micro'] = precision_recall_curve(y_real.ravel(), y_pred.ravel())
    avg_prec['micro'] = average_precision_score(y_real, y_pred, average='micro')

    all_rec = np.unique(np.concatenate([rec[i] for i in range(n_classes)]))
    all_rec = all_rec[np.logical_not(np.isnan(all_rec))]
    # Then interpolate all precision recall curves at these points
    mean_prec = np.zeros_like(all_rec)
    for i in range(n_classes):
        mean_prec += interp(all_rec, np.flip(rec[i], 0), np.flip(prec[i], 0))

    # Finally average it and compute average precision
    mean_prec /= n_classes

    prec['macro'] = np.flip(mean_prec, 0)
    rec['macro'] = np.flip(all_rec, 0)

    avg_prec['macro'] = average_precision_score(y_real, y_pred, average='macro')

    return prec, rec, avg_prec, threshold


def compute_threshold(prec, rec, threshold, n_classes):
    # Find the best cut-off of the average precision curve
    threshold['best'] = {}
    for i in range(n_classes):
        f_score = 0
        b_prec = 0
        for j in range(len(threshold[i])):
            if prec[i][j] == 0 and rec[i][j] == 0:
                continue
            f1_i = 2 * (prec[i][j] * rec[i][j]) / (prec[i][j] + rec[i][j])
            if f_score < f1_i:
                f_score = f1_i
                b_prec = prec[i][j]
                threshold[i][j] = round(100 * threshold[i][j]) / 100
                if threshold[i][j] < 0.01:
                    threshold['best'][i] = 0.01
                elif threshold[i][j] > 0.99:
                    threshold['best'][i] = 0.99
                else:
                    threshold['best'][i] = threshold[i][j]

        print('%i: prec %.2f - fscore %.2f' % (i, b_prec, f_score))
    print('threshold', threshold['best'])

    return threshold


def plot_metric(y_val, x_val, avg, p_name=None, threshold=None):
    # Plot all curves

    n_classes = len(avg) - 2

    fig = plt.figure()
    plt.plot(x_val['micro'], y_val['micro'],
             label='micro-average ({0:0.2f})'
                   ''.format(avg['micro']),
             linestyle='-', color='dodgerblue', linewidth=2, alpha=.9)
    #

    plt.plot(x_val['macro'], y_val['macro'],
             label='macro-average ({0:0.2f})'
                   ''.format(avg['macro']),
             linestyle='-', color='orangered', linewidth=2, alpha=.9)
    # linestyle = ':',

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    x = np.linspace(0.0, 1.0, n_classes)

    rgb = matplotlib.cm.get_cmap(plt.get_cmap('tab20'))(x)[np.newaxis, :, :3][0]

    mlb = load_mlb()

    for i, color in zip(range(n_classes), rgb):
        plt.plot(x_val[i], y_val[i], color=color, linewidth=1, linestyle='--',
                 label='{0} ({1:0.2f})'
                       ''.format(mlb.classes_[i], avg[i]), alpha=.8)

    if threshold is not None:
        plt.plot([0, 1], [threshold, threshold], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate (1 - Specificity)')
    # plt.ylabel('True Positive Rate (Sensitivity)')
    # plt.title('ROC for UniProt classes')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for UniProt classes')
    plt.legend(loc='upper right')
    # plt.show()
    if p_name is None:
        p_name = 'prec_rec_fig'
    fig.savefig(p_name + '.png', bbox_inches='tight')

# from sklearn.datasets import make_multilabel_classification
# from sklearn.preprocessing import minmax_scale
# X, Y = make_multilabel_classification(n_features=11, n_classes=11, n_labels=3, allow_unlabeled=True, random_state=1)
# X = minmax_scale(X)
# y_val, x_val, avg, ts = compute_threshold(Y, X, 11)
# plot_metric(y_val, x_val, avg)
