# coding: utf-8
# %load create_train_test.py
lsize = {}
hsize = {}

cmap = {}
with open('/data/user/teodoro/uniprot/annotation/train_data.tsv') as f:
    for line in f:
        info = line.strip().lower().split('\t')
        cmap[info[1]] = set()
        for i in info[5].split('||'):
            cmap[info[1]].add(i)

with open('/data/user/teodoro/uniprot/dataset/no_large/size_dist/size_dist_lower') as f:
    for line in f:
        info = line.strip().split()
        annot = info[1].split('_')
        if annot[0] not in lsize:
            lsize[annot[0]] = set()
        lsize[annot[0]].add(annot[-1])

with open('/data/user/teodoro/uniprot/dataset/no_large/size_dist/size_dist_upper') as f:
    for line in f:
        info = line.strip().split()
        annot = info[1].split('_')
        if annot[0] not in hsize:
            hsize[annot[0]] = set()
        hsize[annot[0]].add(annot[-1])

from random import shuffle

large_pmids = hsize.keys()
small_pmids = lsize.keys()

large_dist = {}
for pmid in large_pmids:
    for cs in cmap[pmid]:
        if cs not in large_dist:
            large_dist[cs] = 0
        large_dist[cs] += 1

small_dist = {}
for pmid in small_pmids:
    for cs in cmap[pmid]:
        if cs not in small_dist:
            small_dist[cs] = 0
        small_dist[cs] += 1

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pylab as plt

lists = sorted(small_dist.items(), key=lambda item: item[1])
x, y = zip(*lists)  # unpack a list of pairs into two tuples
plt.plot(x, y)
plt.show()

small_pmids = list(lsize.keys())
large_pmids = list(hsize.keys())

hdev = []
pcount = 0
ccount = {}
for i in large_pmids:
    pclass = cmap[i]  # get pmid classes
    add_pmid = True
    for cc in pclass:  # check if pmid does not contain a class well represented
        if cc not in ccount:
            ccount[cc] = 0
        if add_pmid == True and ccount[cc] > 46:
            add_pmid = False
    if add_pmid:  # pmid good to
        for cc in pclass:
            ccount[cc] += 1
        hdev.append(i)
        pcount += 1
    if pcount >= 500:
        break

ldev = []
pcount = 0
ccount = {}
for i in small_pmids:
    pclass = cmap[i]  # get pmid classes
    add_pmid = True
    for cc in pclass:  # check if pmid does not contain a class well represented
        if cc not in ccount:
            ccount[cc] = 0
        if add_pmid == True and ccount[cc] > 46:
            add_pmid = False
    if add_pmid:  # pmid good to
        for cc in pclass:
            ccount[cc] += 1
        ldev.append(i)
        pcount += 1
    if pcount >= 500:
        break

ltest = []
pcount = 0
ccount = {}
shuffle(small_pmids)
for i in small_pmids:
    pclass = cmap[i]  # get pmid classes
    for cc in pclass:  # check if pmid does not contain a class well represented
        if cc not in ccount:
            ccount[cc] = 0
        ccount[cc] += 1
    if i not in ldev:
        ltest.append(i)
        pcount += 1
    if pcount >= 1000:
        break

htest = []
pcount = 0
ccount = {}
shuffle(large_pmids)
for i in large_pmids:
    pclass = cmap[i]  # get pmid classes
    for cc in pclass:  # check if pmid does not contain a class well represented
        if cc not in ccount:
            ccount[cc] = 0
        ccount[cc] += 1
    if i not in hdev:
        htest.append(i)
        pcount += 1
    if pcount >= 1000:
        break

with open('/data/user/teodoro/uniprot/dataset/no_large/size_dist/small_dev', mode='w') as f:
    for i in ldev:
        print(i, file=f)

with open('/data/user/teodoro/uniprot/dataset/no_large/size_dist/large_dev', mode='w') as f:
    for i in hdev:
        print(i, file=f)

with open('/data/user/teodoro/uniprot/dataset/no_large/size_dist/large_test', mode='w') as f:
    for i in htest:
        print(i, file=f)

with open('/data/user/teodoro/uniprot/dataset/no_large/size_dist/small_test', mode='w') as f:
    for i in ltest:
        print(i, file=f)
