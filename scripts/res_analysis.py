import pickle
from pprint import pprint
import math
import matplotlib
import numpy as np
import os
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score

from input.utils import get_class_map

matplotlib.use('Agg')
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from pprint import pprint

from input.utils import load_mlb

plt.style.use('ggplot')

g_scale = 0.5
g_ratio_l = 16.18 * g_scale
g_ratio_a = 10 * g_scale


def get_label_encoder():
    labels = ['ARATH', 'BACSU', 'BOVIN', 'CAEEL', 'CANAL', 'CHICK', 'DANRE', 'DICDI', 'DROME',
              'ECOLI', 'HUMAN', 'MOUSE', 'MYCTU', 'ORYSJ', 'OTHER', 'PIG', 'RAT', 'SCHPO',
              'XENLA', 'YEAST']
    lenc = preprocessing.LabelEncoder()
    lenc.fit(labels)
    return lenc


def get_size_dist():
    size_dist = {}
    fsize = '/data/user/teodoro/uniprot/dataset/pub_data/size_dist/size_dist'

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
    tfile_large = '/data/user/teodoro/uniprot/dataset/pub_data/size_dist/large_test'
    with open(tfile_large) as f:
        for line in f:
            large_test.append(line.strip())

    small_test = []
    tfile_small = '/data/user/teodoro/uniprot/dataset/pub_data/size_dist/small_test'
    with open(tfile_small) as f:
        for line in f:
            small_test.append(line.strip())

    return size_dist, large_test, small_test


def load_test_coll(coll_type, class_name, c):
    tdocs, labels = [], []
    vname = class_name + '_' + c
    if class_name == 'cnn':
        category_map = get_class_map()
        test_file = '/data/user/teodoro/uniprot/results/pub_res/' + coll_type + '/' + vname + '.qid'
        with open(test_file) as f:
            for line in f:
                _doc_id = line.strip()
                tdocs.append(_doc_id)
                labels.append(category_map[_doc_id])
        f.close()
        labels = np.asarray(labels)
    else:
        test_file = '/data/user/teodoro/uniprot/dataset/pub_data/processed/' + coll_type + '/test_docs.csv'
        with open(test_file) as f:
            for line in f:
                tdocs.append(line.strip())
        f.close()

        test_labels = '/data/user/teodoro/uniprot/dataset/pub_data/processed/' + coll_type + '/test_labels.csv'
        labels = np.loadtxt(test_labels, delimiter=',')

    test_pred = '/data/user/teodoro/uniprot/results/pub_res/' + coll_type + '/' + vname + '.res'
    y_pred = np.loadtxt(test_pred, delimiter=',')

    return tdocs, labels, y_pred


def get_stats(test_pmids, size_dist, test_labels, test_pred, small_pmids, large_pmids):
    lenc = get_label_encoder()

    mlb = load_mlb()

    precision = []
    org_cat = {}
    ap_by_size, ap_by_size_small, ap_by_size_large = (), (), ()
    ap_by_org, ap_by_prot, ap_by_org_mean, ap_by_prot_mean = {}, {}, {}, {}
    for i in range(len(test_pmids)):
        pmid, prot, org = test_pmids[i].split('_', 2)
        if org not in lenc.classes_:
            org = 'OTHER'

        # qprec = precision_score(test_labels[i], test_pred[i], average='binary')
        qprec = precision_score(np.array([test_labels[i]]), np.array([test_pred[i]]), average='micro')
        precision.append(qprec)
        if org not in ap_by_org:
            ap_by_org[org] = []
            org_cat[org] = {}

        ap_by_org[org].append(qprec)

        for l in mlb.inverse_transform(np.array([test_labels[i]]))[0]:
            if l == "unclassified":
                l = "miscellaneous"

            if l not in org_cat[org]:
                org_cat[org][l] = 0
            org_cat[org][l] += 1

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

    return precision, ap_by_size, ap_by_size_small, ap_by_size_large, ap_by_org, ap_by_prot, ap_by_org_mean, ap_by_prot_mean, org_cat


def get_norm_size_stats(size_x, prec_y, div=1):
    n_prec = {}
    rn_size, rn_prec, rn_vol, rn_col = [], [], [], []
    size_max = max(size_x)
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
            n_prec[n_size][prec_y_i]['c'] = prec_y_i * math.log2(size_max / p_size)
            # n_prec[n_size][prec_y_i]['c'] = -1 * (prec_y_i * math.log(p_size / size_max))
            # n_prec[n_size][prec_y_i]['c'] = prec_y_i/(p_size / size_max)
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

def plot_histogram_pmids(values):
    print('------------------')
    print('prec by PROTEIN')
    # pprint(prec_prot_m)

    fig = plt.figure(figsize=(g_ratio_l, g_ratio_a))
    plt.hist(values, 50, facecolor='g', alpha=0.75)
    plt.xlabel('#Count')
    plt.ylabel('#Category set / Publication')

    fig.savefig(class_name + '_pmid.png', bbox_inches='tight', dpi=300)

def plot_histogram(class_name, prec_prot_m):
    print('------------------')
    print('prec by PROTEIN')
    # pprint(prec_prot_m)

    fig = plt.figure(figsize=(g_ratio_l, g_ratio_a))
    plt.hist(list(prec_prot_m.values()), bins=9)
    plt.xlabel('Precision (Micro)')
    plt.ylabel('Count #Protein')

    fig.savefig(class_name + '_prot.png', bbox_inches='tight', dpi=300)


def plot_organism(class_name, prec_org_m, org_cat):
    print('------------------')
    print('prec by ORGANISM')
    pprint(prec_org_m)

    fig = plt.figure(figsize=(g_ratio_l, g_ratio_a))
    sorted_cats = []
    for i in sorted(list(prec_org_m.keys())):
        if i != 'OTHER':
            sorted_cats.append(i)
    sorted_cats.append('OTHER')

    series = []
    new_labels = []
    for label in load_mlb().classes_:
        if label == "unclassified":
            label = "miscellaneous"
        new_labels.append(label)

    for label in new_labels:
        y_cats = []
        for i in sorted_cats:
            cat_total = sum([v for k, v in org_cat[i].items()])
            if label in org_cat[i]:
                y_cats.append(prec_org_m[i] * (org_cat[i][label] / cat_total))
            else:
                y_cats.append(0)
        series.append(y_cats)

    x_cats = list(range(len(sorted_cats)))
    x = np.linspace(0.0, 1.0, len(sorted_cats))
    rgb = matplotlib.cm.get_cmap(plt.get_cmap('tab20'))(x)[np.newaxis, :, :3][0]

    plots = []
    y_offset = np.array([0] * len(sorted_cats))
    for i, serie in enumerate(series):
        p = plt.bar(x_cats, serie, color=rgb[i], bottom=y_offset, alpha=.7)
        y_offset = y_offset + np.array(serie)
        plots.append(p[0])

    plt.xticks(x_cats, sorted_cats, rotation='vertical')
    plt.axhline(np.mean([v for k, v in prec_org_m.items()]), ls='-.', color='black', linewidth=1)
    plt.margins(0.01)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel('Organism')
    plt.ylabel('Precision (Micro)')
    plt.legend(plots, new_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    fig.savefig(class_name + '_org.png', bbox_inches='tight', dpi=300)


def plot_size(class_name, prec_size, prec_size_s=None, prec_size_l=None):
    print('------------------')
    print('prec by SIZE')

    size_x, prec_y = zip(*prec_size)
    rn_size, rn_prec, rn_vol, rn_col = get_norm_size_stats(size_x, prec_y)

    fig = plt.figure(figsize=(g_ratio_l, g_ratio_a))
    plt.scatter(rn_size, rn_prec, c=rn_col, s=rn_vol, alpha=0.5)
    plt.xlabel('Size (KB)')
    plt.ylabel('Precision (Micro)')
    plt.colorbar(label='precision * log$_{2}$(max_size/size)')

    for docs in [10, 100, 1000]:
        plt.scatter([], [], c='k', alpha=0.3, s=docs, label=str(docs))

    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Publications')
    fig.savefig(class_name + '_size.png', bbox_inches='tight', dpi=300)

    print("correlation prec <-> size:", np.corrcoef(rn_size, rn_prec))

    # small_size_x, small_prec_y = zip(*prec_size_s)
    # rn_size, rn_prec, rn_vol, rn_col = get_norm_size_stats(small_size_x, small_prec_y)
    #
    # fig = plt.figure(figsize=(g_ratio_l, g_ratio_a))
    # plt.scatter(rn_size, rn_prec, c=rn_col, s=rn_vol, alpha=0.5)
    # plt.xlabel('Size (KB)')
    # plt.ylabel('Precision')
    #
    # fig.savefig(class_name + '_small_size.png', bbox_inches='tight', dpi=300)
    #
    # large_size_x, large_prec_y = zip(*prec_size_l)
    # rn_size, rn_prec, rn_vol, rn_col = get_norm_size_stats(large_size_x, large_prec_y, div=10)
    #
    # fig = plt.figure(figsize=(g_ratio_l, g_ratio_a))
    # plt.scatter(rn_size, rn_prec, c=rn_col, s=rn_vol, alpha=0.5)
    # plt.xlabel('Size (KB)')
    # plt.ylabel('Precision')
    #
    # fig.savefig(class_name + '_large_size.png', bbox_inches='tight', dpi=300)


def save_precision(class_name, precision):
    with open(class_name + '_prec.res', mode='w') as prec_file:
        for prec in precision:
            print(prec, file=prec_file)
    prec_file.close()

def list_pmids_multiclass(test_pmids, test_labels, test_pred):
    mlb = load_mlb()
    pmids = {}
    for i, doc in enumerate(test_pmids):
        pmid, _ = doc.split("_", maxsplit=1)
        if pmid not in pmids:
            pmids[pmid] = {"count": 0, "categories": [], "docs": {}, "score": []}
        pmids[pmid]["count"] += 1

        score = f1_score(np.array([test_labels[i]]), np.array([test_pred[i]]), average="micro")
        pmids[pmid]["score"].append(score)

        all_labels = mlb.inverse_transform(np.array([test_labels[i]]))[0]
        label = set(all_labels)
        pmids[pmid]["docs"][doc] = label
        label_in = False
        for cats in pmids[pmid]["categories"]:
            if label == cats:
                label_in = True
        if not label_in:
            pmids[pmid]["categories"].append(label)

    count_class = 0
    pmid_annotations = []

    max_categories = 0
    pmid_selected = None
    for pmid in pmids.keys():
        if pmid_selected != pmid and len(pmids[pmid]["categories"]) == 5 and sum(pmids[pmid]["score"])/len(pmids[pmid]["score"]) > 0.7 \
                and pmid not in ["15326186", "15326186", "25023281", "23106124",
                                 "23072806", "22925462", "20510931", "11283612", "11847227", "22948820", "15326186"]:
            # and sum(pmids[pmid]["score"])/len(pmids[pmid]["score"]) <= 0.6:
            max_categories = len(pmids[pmid]["categories"])
            pmid_selected = pmid

        pmid_annotations.append(pmids[pmid]["count"])
        if pmids[pmid]["count"] >= 2:
            count_class += 1

    print(pmid_selected, pmids[pmid_selected]["count"])
    print(pmids[pmid_selected]["categories"])
    print(pmids[pmid_selected]["docs"])

    print("total pmids", len(pmids))
    print("total pmids multiple annotation", count_class)

    return pmid_annotations
# class_name, c, coll_type = 'svm', '4.641588833612782e-05', 'tag'
# class_name, c, coll_type = 'svm', '0.000774263682681127', 'tag'
# class_name, c, coll_type = 'cnn', '3.162277660168379e-06', 'no_tag'
class_name, c, coll_type = 'cnn', '0.03162277660168379', 'tag'

print('loading stats')
res_file = class_name + '_res_stats.pkl'
res_stat = {}
test_labels, test_pred = None, None
if not os.path.exists(res_file):
    size_dist, large_pmids, small_pmids = get_size_dist()
    test_pmids, test_labels, test_pred = load_test_coll(coll_type, class_name, c)
    precision, prec_size, prec_size_s, prec_size_l, prec_org, prec_prot, prec_org_m, prec_prot_m, org_cat \
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
    res_stat['org_cat'] = org_cat

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
    prec_org, prec_prot, prec_org_m, prec_prot_m, org_cat = res_stat['prec_org'], res_stat['prec_prot'], res_stat[
        'prec_org_m'], res_stat['prec_prot_m'], res_stat['org_cat']

print('PRECISION - RECALL - FSCORE')
print('prec micro', precision_score(test_labels, test_pred, average='micro'))
print('prec macro', precision_score(test_labels, test_pred, average='macro'))
print('rec micro', recall_score(test_labels, test_pred, average='micro'))
print('rec macro', recall_score(test_labels, test_pred, average='macro'))
print('f1 micro', f1_score(test_labels, test_pred, average='micro'))
print('f1 macro', f1_score(test_labels, test_pred, average='macro'))

# save_precision(class_name, precision)
# plot_histogram(class_name, prec_prot_m)
# plot_organism(class_name, prec_org_m, org_cat)
plot_size(class_name, prec_size, prec_size_s=prec_size_s, prec_size_l=prec_size_l)
# pmid_annotations = list_pmids_multiclass(test_pmids, test_labels, test_pred)

# plot_histogram_pmids(pmid_annotations)

# tag_res = [
#     [1.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000]
# ]
#
# no_tag_res = [
#     [1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000]
# ]
#
# mlb = load_mlb()
#
# tag = mlb.inverse_transform(np.array(tag_res))
# pprint(tag)
#
# notag = mlb.inverse_transform(np.array(no_tag_res))
# pprint(notag)
