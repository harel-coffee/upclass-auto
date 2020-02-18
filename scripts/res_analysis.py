import pickle

import math
import matplotlib
import numpy as np
import os
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score

# matplotlib.use('Agg')
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from pprint import pprint

plt.style.use('ggplot')


def get_label_encoder():
    labels = ['ARATH', 'BACSU', 'BOVIN', 'CAEEL', 'CANAL', 'CHICK', 'DANRE', 'DICDI', 'DROME',
              'ECOLI', 'HUMAN', 'MOUSE', 'MYCTU', 'ORYSJ', 'OTHER', 'PIG', 'RAT', 'SCHPO',
              'XENLA', 'YEAST']
    lenc = preprocessing.LabelEncoder()
    lenc.fit(labels)
    return lenc


def get_size_dist():
    size_dist = {}
    fsize = '/data/user/teodoro/uniprot/dataset/no_large/size_dist/size_dist'

    with open(fsize) as f:
        for line in f:
            size, doc_tag = line.strip().split()
            if 'K' in size:
                size = float(size.replace('K', '')) * 1024
            else:
                size = float(size)
            pmid, prot = doc_tag.split('_', 1)
            size_dist[pmid] = size / 1024

    large_test = []
    tfile_large = '/data/user/teodoro/uniprot/dataset/no_large/size_dist/large_test'
    with open(tfile_large) as f:
        for line in f:
            large_test.append(line.strip())

    small_test = []
    tfile_small = '/data/user/teodoro/uniprot/dataset/no_large/size_dist/small_test'
    with open(tfile_small) as f:
        for line in f:
            small_test.append(line.strip())

    return size_dist, large_test, small_test


def load_test_coll(coll_type):
    tdocs = []
    test_file = '/data/user/teodoro/uniprot/dataset/no_large/processed/' + coll_type + '/test_docs.csv'
    with open(test_file) as f:
        for line in f:
            tdocs.append(line.strip())
    f.close()

    test_labels = '/data/user/teodoro/uniprot/dataset/no_large/processed/' + coll_type + '/test_labels.csv'
    labels = np.loadtxt(test_labels, delimiter=',')

    vname = class_name + '_' + c

    test_pred = '/data/user/teodoro/uniprot/results/no_large/' + coll_type + '/' + vname + '.res'
    y_pred = np.loadtxt(test_pred, delimiter=',')

    return tdocs, labels, y_pred


def get_stats(test_pmids, size_dist, test_labels, test_pred, small_pmids, large_pmids):
    lenc = get_label_encoder()

    precision = []
    ap_by_size, ap_by_size_small, ap_by_size_large = (), (), ()
    ap_by_org, ap_by_prot, ap_by_org_mean, ap_by_prot_mean = {}, {}, {}, {}
    for i in range(len(test_pmids)):
        pmid, prot, org = test_pmids[i].split('_', 2)
        if org not in lenc.classes_:
            org = 'OTHER'
        qprec = precision_score(test_labels[i], test_pred[i], average='binary')
        precision.append(qprec)
        if org not in ap_by_org:
            ap_by_org[org] = []
        ap_by_org[org].append(qprec)

        if prot not in ap_by_prot:
            ap_by_prot[prot] = []
        ap_by_prot[prot].append(qprec)

        size = size_dist[pmid]
        ap_by_size = ap_by_size + ((size, qprec),)
        if pmid in small_pmids:
            ap_by_size_small = ap_by_size_small + ((size, qprec),)
        elif pmid in large_pmids:
            ap_by_size_large = ap_by_size_large + ((size, qprec),)

    for k, v in ap_by_org.items():
        ap_by_org_mean[k] = sum(v) / len(v)

    for k, v in ap_by_prot.items():
        ap_by_prot_mean[k] = sum(v) / len(v)

    return precision, ap_by_size, ap_by_size_small, ap_by_size_large, ap_by_org, ap_by_prot, ap_by_org_mean, ap_by_prot_mean


def get_norm_size_stats(size_x, prec_y, div=1):
    n_prec = {}
    rn_size, rn_prec, rn_vol, rn_col = [], [], [], []
    size_max = 100
    for i in range(len(size_x)):
        n_size = round(size_x[i] / div) * div
        prec_y_i = prec_y[i]
        if n_size not in n_prec:
            n_prec[n_size] = {}

        if prec_y_i not in n_prec[n_size]:
            n_prec[n_size][prec_y_i] = {}
            n_prec[n_size][prec_y_i]['v'] = 0

            p_size = n_size
            if n_size == 0:
                p_size = size_x[i]

            if p_size > size_max:
                print(p_size)
                size_max = p_size
            n_prec[n_size][prec_y_i]['c'] = -1 * (prec_y_i * math.log(p_size / 141))
        n_prec[n_size][prec_y_i]['v'] += 1

    for i in sorted(n_prec.keys()):
        for j in n_prec[i].keys():
            vol_ij = n_prec[i][j]['v']
            col_ij = n_prec[i][j]['c']

            rn_size.append(i)
            rn_prec.append(j)
            rn_vol.append(vol_ij)
            rn_col.append(col_ij)

    return rn_size, rn_prec, rn_vol, rn_col


# class_name, c, coll_type = 'svm', '0.0001', 'no_tag'
# class_name, c, coll_type = 'svm', '0.000774263682681127', 'tag'
# class_name, c, coll_type = 'cnn', '3.162277660168379e-06', 'no_tag'
class_name, c, coll_type = 'cnn', '1e-05', 'tag'

res_file = class_name + '_' + c + '_res_stats.pkl'
res_stat = {}
if not os.path.exists(res_file):
    size_dist, large_pmids, small_pmids = get_size_dist()
    test_pmids, test_labels, test_pred = load_test_coll(coll_type)
    precision, prec_size, prec_size_s, prec_size_l, prec_org, prec_prot, prec_org_m, prec_prot_m \
        = get_stats(test_pmids, size_dist, test_labels, test_pred, small_pmids, large_pmids)

    res_stat['size_dist'] = size_dist
    res_stat['large_pmids'] = large_pmids
    res_stat['small_pmids'] = small_pmids

    res_stat['test_pmids'] = test_pmids
    res_stat['test_labels'] = test_labels
    res_stat['test_pred'] = test_pred

    res_stat['precision'] = precision
    res_stat['prec_size'] = prec_size
    res_stat['prec_size_s'] = prec_size_s
    res_stat['prec_size_l'] = prec_size_l

    res_stat['prec_org'] = prec_org
    res_stat['prec_prot'] = prec_prot
    res_stat['prec_org_m'] = prec_org_m
    res_stat['prec_prot_m'] = prec_prot_m

    with open(res_file, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(res_stat, f, pickle.HIGHEST_PROTOCOL)
    f.close()
else:
    with open(res_file, 'rb') as f:
        res_stat = pickle.load(f)
    f.close()

    size_dist, large_pmids, small_pmids = res_stat['size_dist'], res_stat['large_pmids'], res_stat['small_pmids']
    test_pmids, test_labels, test_pred = res_stat['test_pmids'], res_stat['test_labels'], res_stat['test_pred']
    precision, prec_size, prec_size_s, prec_size_l = res_stat['precision'], res_stat['prec_size'], res_stat[
        'prec_size_s'], res_stat['prec_size_l']
    prec_org, prec_prot, prec_org_m, prec_prot_m = res_stat['prec_org'], res_stat['prec_prot'], res_stat['prec_org_m'], \
                                                   res_stat['prec_prot_m']

print('------------------')
print('PRECISION - RECALL - FSCORE')

with open(class_name + '_' + c + '_prec.res', mode='w') as prec_file:
    for prec in precision:
        print(prec, file=prec_file)
prec_file.close()

print('prec micro', precision_score(test_labels, test_pred, average='micro'))
print('prec macro', precision_score(test_labels, test_pred, average='macro'))
print('rec micro', recall_score(test_labels, test_pred, average='micro'))
print('rec macro', recall_score(test_labels, test_pred, average='macro'))
print('f1 micro', f1_score(test_labels, test_pred, average='micro'))
print('f1 macro', f1_score(test_labels, test_pred, average='macro'))

g_scale = 0.5
g_ratio_l = 16.18 * g_scale
g_ratio_a = 10 * g_scale

print('------------------')
print('prec by PROTEIN')
# #pprint(ap_by_prot_mean)
#
# fig = plt.figure()
# plt.hist(list(prec_prot_m.values()), bins=9)
# plt.xlabel('Precision (Micro)')
# plt.ylabel('Count')
# plt.show()
# #fig.savefig(class_name + '_prot.png', bbox_inches='tight')

print('')
print('------------------')
print('prec by ORGANISM')
pprint(prec_org_m)

fig = plt.figure(figsize=(g_ratio_l, g_ratio_a))
sorted_cats = sorted(list(prec_org_m.keys()))
cats = []
y_cats = []
for i in sorted_cats:
    if i != 'OTHER':
        cats.append(i)
        y_cats.append(prec_org_m[i])

cats.append('OTHER')
y_cats.append(prec_org_m['OTHER'])

x_cats = list(range(len(cats)))

plt.bar(x_cats, y_cats,
        # color=[v['color'] for v in list(plt.rcParams['axes.prop_cycle'])],
        color=list(plt.rcParams['axes.prop_cycle'])[1]['color'],
        #        cmap=plt.cm.get_cmap('tab20'),
        alpha=.7)

plt.xticks(x_cats, cats, rotation='vertical')
plt.axhline(np.mean(y_cats), ls='-.', color='black', linewidth=1)
plt.margins(0.01)
plt.subplots_adjust(bottom=0.15)
plt.xlabel('Organism')
plt.ylabel('Precision (Micro)')

# plt.show()
fig.savefig(class_name + '_' + c + '_org.pdf', bbox_inches='tight')
# plt.close()

print('')
print('------------------')
print('prec by SIZE')

size_x, prec_y = zip(*prec_size)

rn_size, rn_prec, rn_vol, rn_col = get_norm_size_stats(size_x, prec_y)

fig = plt.figure(figsize=(g_ratio_l, g_ratio_a))
plt.scatter(rn_size, rn_prec, c=rn_col, s=rn_vol, alpha=0.5)
plt.xlabel('Size (KB)')
plt.ylabel('Precision (Micro)')
# plt.show()
fig.savefig(class_name + '_' + c + '_size.pdf', bbox_inches='tight')

# small_size_x, small_prec_y = zip(*prec_size_s)
#
# rn_size, rn_prec, rn_vol, rn_col = get_norm_size_stats(small_size_x, small_prec_y)
#
# fig = plt.figure(figsize=(7, 7))
# plt.scatter(rn_size, rn_prec, c=rn_col, s=rn_vol, alpha=0.5)
# plt.xlabel('Size (KB)')
# plt.ylabel('Precision')
# plt.show()
# #fig.savefig(class_name + '_small_size.png', bbox_inches='tight')
#
#
# large_size_x, large_prec_y = zip(*prec_size_l)
# rn_size, rn_prec, rn_vol, rn_col = get_norm_size_stats(large_size_x, large_prec_y, div=10)
#
# fig = plt.figure(figsize=(7, 7))
# plt.scatter(rn_size, rn_prec, c=rn_col, s=rn_vol, alpha=0.5)
# plt.xlabel('Size (KB)')
# plt.ylabel('Precision')
# plt.show()
# #fig.savefig(class_name + '_large_size.png', bbox_inches='tight')
