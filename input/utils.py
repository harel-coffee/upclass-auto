import os
import pickle
from random import shuffle

import numpy as np
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.preprocessing import MultiLabelBinarizer


def load_mlb():
    mlb = MultiLabelBinarizer()
    mlb.fit([['ptm/processing', 'family & domains', 'interaction', 'expression', 'sequences', 'subcellular location',
              'unclassified', 'pathology & biotech', 'names', 'structure', 'function']])
    return mlb


def get_class_map():
    cmap = {}
    mlb = load_mlb()
    source_dir = '/data/user/teodoro/uniprot/annotation'
    source_tsv = source_dir + '/train_data.tsv'
    source_pkl = source_dir + '/train_data.pkl'

    if os.path.exists(source_pkl):
        with open(source_pkl, 'rb') as f:
            cmap = pickle.load(f)
        f.close()
    else:
        with open(source_tsv) as f:
            for line in f:
                info = line.strip().split('\t')
                f_tag = info[1] + '_' + info[2]
                cmap[f_tag] = mlb.transform([[v.lower() for v in info[5].split('||')]])[0]
        f.close()

        with open(source_pkl, 'wb') as f:
            pickle.dump(cmap, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    return cmap


def get_ratio(u_int, counts_int):
    ratio = {}
    p_idx = None
    for k in range(len(u_int)):
        c_min = 1000000
        for i in range(len(u_int)):
            if c_min > counts_int[i] and u_int[i] not in ratio:
                c_min = counts_int[i]
                i_idx = i
        val = counts_int[i_idx]
        if p_idx is not None and val > 1.25 * ratio[u_int[p_idx]]:
            val = int(1.25 * ratio[u_int[p_idx]])
        ratio[u_int[i_idx]] = val
        p_idx = i_idx
    print(ratio)
    return ratio


def under_sample(X, y):
    # s = '{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}'
    s = ''
    for i in range(y.shape[1]):
        s += '{' + str(i) + '}'

    ratio = {64: 233, 512: 291, 2: 363, 1: 707, 16: 453, 32: 566, 8: 2152, 4: 1103, 128: 1378, 256: 1722, 1024: 883}

    Y = np.asarray(y, dtype=int)
    Y_int = np.asarray([int(s.format(*Y[i]), 2) for i in range(len(Y))])
    ind_class = np.where(
        (Y_int[:] == 1024) + (Y_int[:] == 512) + (Y_int[:] == 256) + (Y_int[:] == 128) + (Y_int[:] == 64) + (
                Y_int[:] == 32) + (Y_int[:] == 16) + (Y_int[:] == 8) + (Y_int[:] == 4) + (Y_int[:] == 2) + (
                Y_int[:] == 1))

    print(ind_class)
    print(ind_class[0])

    X_main = X[ind_class]
    Y_main = Y[ind_class]
    Y_main_int = Y_int[ind_class]

    # print(X_main[0])
    print('X_main', len(X_main))

    # print(Y_main[0])
    print('Y_main', len(Y_main))

    # print(Y_main_int[0])
    print('Y_main_int', len(Y_main_int))

    rus = RandomUnderSampler(ratio=ratio, return_indices=True)
    X_main_res, _, idx_resampled = rus.fit_sample(X_main, Y_main_int)
    Y_main_res = Y_main[idx_resampled]

    print('X_main_res', len(X_main_res))
    print('Y_main_res', len(Y_main_res))
    print('Y_main_res', Y_main_res[0])

    ind_multi_class = np.where(
        (Y_int[:] != 1024) * (Y_int[:] != 512) * (Y_int[:] != 256) * (Y_int[:] != 128) * (Y_int[:] != 64) * (
                Y_int[:] != 32) * (Y_int[:] != 16) * (Y_int[:] != 8) * (Y_int[:] != 4) * (Y_int[:] != 2) * (
                Y_int[:] != 1))

    X_comp = X[ind_multi_class]
    Y_comp = Y[ind_multi_class]
    Y_comp_int = Y_int[ind_multi_class]

    yci_class, yci_count = np.unique(Y_comp_int, return_counts=True)
    yci_class_filter = yci_class[np.where(yci_count[:] >= 263)]
    yci_idx_filter = (np.asarray([i for i in range(Y_comp_int.shape[0]) if Y_comp_int[i] in yci_class_filter]),)

    X_comp = X_comp[yci_idx_filter]
    Y_comp = Y_comp[yci_idx_filter]
    Y_comp_int = Y_comp_int[yci_idx_filter]

    # print(X_comp[0])
    print('X_comp', len(X_comp))

    # print(Y_comp[0])
    print('Y_comp', len(Y_comp))

    # print(Y_comp_int[0])
    print('Y_comp_int', len(Y_comp_int))

    rus = RandomUnderSampler(return_indices=True)
    X_comp_res, _, idx_resampled = rus.fit_sample(X_comp, Y_comp_int)
    Y_comp_res = Y_comp[idx_resampled]

    print('X_comp_res', len(X_comp_res))
    print('Y_comp_res', len(Y_comp_res))
    print('Y_comp_res', Y_comp_res[0])

    X_res = np.vstack((X_main_res, X_comp_res))
    Y_res = np.vstack((Y_main_res, Y_comp_res))

    ix = list(range(len(Y_res)))
    shuffle(ix)
    X_res = X_res[ix]
    Y_res = Y_res[ix]
    return Y_res, X_res


def resample(X, y):
    # s = '{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}'
    s = ''
    for i in range(y.shape[1]):
        s += '{' + str(i) + '}'

    Y = np.asarray(y, dtype=int)
    Y_int = np.asarray([int(s.format(*Y[i]), 2) for i in range(len(Y))])

    yci_class, yci_count = np.unique(Y_int, return_counts=True)
    yci_class_filter = yci_class[np.where(yci_count[:] >= 50)]
    yci_idx_filter = (np.asarray([i for i in range(Y_int.shape[0]) if Y_int[i] in yci_class_filter]),)

    X_filt = X[yci_idx_filter]
    Y_filt = Y[yci_idx_filter]
    Y_int_filt = Y_int[yci_idx_filter]

    print('X_comp', len(X_filt))
    print('Y_comp', len(Y_filt))
    print('Y_comp_int', len(Y_int_filt))

    tls = TomekLinks(return_indices=True, n_jobs=20)
    X_res, _, idx_resampled = tls.fit_sample(X_filt, Y_int_filt)
    Y_res = Y_filt[idx_resampled]

    print('X_comp_res', len(X_res))
    print('Y_comp_res', len(Y_res))
    print('Y_comp_res', Y_res[0])

    ix = list(range(len(Y_res)))
    shuffle(ix)
    X_res = X_res[ix]
    Y_res = Y_res[ix]
    return Y_res, X_res


def order_test_set(test_set, query_doc_file):
    query_doc = []
    with open(query_doc_file) as f:
        for line in f:
            query_doc.append(line.strip())

    data_test_set = [i for i in test_set]

    if len(data_test_set) != len(query_doc):
        print('error: length test set ({}) differs from query set ({})'.format(len(data_test_set), len(query_doc)))
        exit()

    new_test_set = [() for i in range(len(data_test_set))]
    for i in data_test_set:
        doc_id = i[0]
        index = query_doc.index(doc_id)
        new_test_set[index] = i
    return new_test_set
