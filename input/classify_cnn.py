#!/usr/bin/env python
from __future__ import print_function

import pickle

import numpy as np
import os
from random import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score

from upclass.uniprot.classifier.dataset import TaggedDataset, SentenceDataset
from upclass.uniprot.classifier.model import CoreClassifier
from upclass.uniprot.input import preprocces
from upclass.uniprot.input.article import extract_tags
from upclass.uniprot.input.regressors import get_label_set
from upclass.uniprot.input.utils import load_mlb


def get_features(accession, pmid, articles=None, accessions=None, mtype='tag'):
    pmid = str(pmid)

    article_dict = None
    if pmid not in articles:
        article_dict = preprocces.get_article_dict(pmid, accession[0])
        articles[pmid] = article_dict
    else:
        article_dict = articles[pmid]

    tag_info, accessions = preprocces.get_tag_info(accession, accessions)

    feat_doc = get_text(article_dict, tag_info, mtype)

    return feat_doc, tag_info, articles, accessions


def get_text(article_dict, tag_info, mtype='tag'):
    pre_text, _ = extract_tags(article_dict, tag_info)

    protein = list(tag_info.keys())[0]
    doc_id = pmid + '_' + protein

    feat_doc = []
    if mtype == 'tag':
        feat_doc = ['', '']
        td = TaggedDataset()
        for doc in td.get_content_from_dict(doc_id, pre_text[protein]):
            if '_IN_' in doc.tags[0]:
                feat_doc[0] = doc.words
            elif '_OUT_' in doc.tags[0]:
                feat_doc[1] = doc.words
            else:
                print('error: wrong doc tag', doc.tags[0])
    else:
        td = SentenceDataset()
        for doc in td.get_content_from_dict(doc_id, article_dict):
            feat_doc.append(doc.words)

    return feat_doc


test_source = 'ceci'
# test_source = 'new_sp'
# test_source = 'ceci_new'
# test_source = 'xaa'

mtype = 'tag'

test_set = None
eval = False
if test_source == 'ceci':
    test_set = '/data/user/teodoro/uniprot/annotation/new_annotation/test_set_ceci.tsv'
    eval = True
elif test_source == 'ceci_new':
    test_set = '/data/user/teodoro/uniprot/annotation/new_annotation/test_set_ceci_new.tsv'
    eval = False
elif test_source == 'ceci_new':
    test_set = '/data/user/teodoro/uniprot/annotation/new_annotation/test_set.tsv'
    eval = True
elif test_source.startswith('x'):
    test_set = '/data/user/teodoro/uniprot/annotation/new_annotation/' + test_source
    eval = False

mlb = load_mlb()
articles = {}
accessions = {}
art_file = test_source + '_article_dict.pkl'
acc_file = test_source + '_accession_dict.pkl'

if os.path.exists(art_file):
    with open(art_file, 'rb') as f:
        articles = pickle.load(f)
    f.close()

if os.path.exists(acc_file):
    with open(acc_file, 'rb') as f:
        accessions = pickle.load(f)
    f.close()

count = 0
dataset = ()
labels = {}
docs = []

docs_rep = {}
doc_rep_max = 10

with open(test_set) as test_file:
    for line in test_file:
        if eval:
            (accession_all, pmid, target) = line.strip().split('\t')
        else:
            (accession_all, pmid) = line.strip().split('\t')
        accession = accession_all.split('||')
        shuffle(accession)
        # print('processing %s %s' % (accession, pmid))
        try:
            doc_id = pmid + '_' + accession[0]
            if doc_rep_max > 0 and pmid in docs_rep and docs_rep[pmid] > 10:
                continue
            if not eval or doc_id not in labels:
                text, prot_info, articles, accessions = get_features(accession, pmid, articles=articles,
                                                                     accessions=accessions,
                                                                     mtype=mtype)
                docs.append(doc_id)
                if mtype == 'tag':
                    dataset += ((doc_id, text[0], text[1]),)
                else:
                    dataset += ((doc_id, text[0]),)

                if eval:
                    mt = [t.lower() for t in target.split('||')]
                    #            mt = [t.lower() for t in target]
                    bin_labels = mlb.transform([mt])[0]

                    labels[doc_id] = bin_labels

                if pmid not in docs_rep:
                    docs_rep[pmid] = 0
                docs_rep[pmid] += 1

        except Exception as e:
            print('could not process', pmid, accession)
            print(str(e))
        count += 1
        if count % 1000 == 0:
            print(count, 'items processed')

if not os.path.exists(art_file):
    with open(art_file, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(articles, f, pickle.HIGHEST_PROTOCOL)
    f.close()

if not os.path.exists(acc_file):
    with open(acc_file, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(accessions, f, pickle.HIGHEST_PROTOCOL)
    f.close()

cnn_model_file = '/data/user/teodoro/uniprot/results/no_large/tag/cnn_1e-05.pkl'

nclasses = 11
max_doc_length = 1500

cnn = CoreClassifier('cnn')
model = cnn.load(cnn_model_file)

model.predict_proba(dataset, [])

y_pred = model.predictions
y_queries = model.queries

for i in range(nclasses):
    y_pred[y_pred[:, i] >= model.best_params['threshold'][i], i] = 1
    y_pred[y_pred[:, i] < model.best_params['threshold'][i], i] = 0

if eval:
    _, labels = get_label_set(y_queries, [], [], labels)
    labels = np.asarray(labels, dtype=int)
    print('prec_micro', precision_score(labels, y_pred, average='micro'))
    print('rec_micro', recall_score(labels, y_pred, average='micro'))
    print('fscore_micro', f1_score(labels, y_pred, average='micro'))

y_pred_lab = mlb.inverse_transform(y_pred)

for i in range(len(y_queries)):
    (pmid, accession) = y_queries[i].split('_')
    label = ']['.join(y_pred_lab[i])
    label = '[' + label
    label = label + ']'
    print(accession, pmid, label)
