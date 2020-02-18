#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Douglas Teodoro <dhteodoro@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s -s MODEL_FILE -d OUTPUT_MODEL_DIR -l LANGUAGE [-n WORKERS] [-p]

Train doc2vec model from a flat text corpus.

This actually creates three files:

* `OUTPUT_PREFIX_wordids.txt`: mapping between words and their integer ids
* `OUTPUT_PREFIX_bow.mm`: bag-of-words (word counts) representation, in
  Matrix Matrix format
* `OUTPUT_PREFIX_tfidf.mm`: TF-IDF representation
* `OUTPUT_PREFIX.tfidf_model`: TF-IDF model dump

Example: python -m uniprot.embedding.train_model -s source_dir -d dest_dir -n 16
"""

from __future__ import print_function

import logging
import pickle
import sys
from collections import defaultdict

import numpy as np
import os.path
from copy import deepcopy
from keras import backend as K
from random import seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from upclass.uniprot import classifier
from upclass.uniprot.classifier.cnn import CNN1D
from upclass.uniprot.classifier.multi_label_curve import plot_metric, compute_metrics, compute_threshold
from upclass.uniprot.input.utils import load_mlb

UNIPROT_CLASSES = (
    (1, 'Expression'),
    (2, 'Family & Domains'),
    (3, 'Function'),
    (4, 'Interaction'),
    (5, 'Names'),
    (6, 'Pathology & Biotech'),
    (7, 'PTM/processing'),
    (8, 'Sequences'),
    (9, 'Structure'),
    (10, 'Subcellular location'),
    (11, 'Miscellaneous'),
)

LABEL_MAP = {i[1].lower(): i[1] for i in UNIPROT_CLASSES}

seed(a=41)

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)


def limit_mem():
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def clear_mem():
    sess = K.get_session()
    sess.close()
    limit_mem()
    return


class prettyfloat(float):
    def __repr__(self):
        return "%0.6f" % self


def norm_score(self, score):
    sum_sco = 0
    norm_score = {}
    for k, v in score.items():
        sum_sco += score[k]

    for k, v in score.items():
        if sum_sco != 0:
            norm_score[k] = v / sum_sco
        else:
            norm_score[k] = 0
    return norm_score


class CoreClassifier(object):
    def __init__(self, name, n_workers=8):
        self.name = name
        self.n_workers = n_workers

        self.best_params = {}
        self.init_best_params()

        self.C = None

        self.queries = None
        self.predictor = None
        self.predictions = None
        self.best_predictor = None
        self.best_results = None

        self.scaler = None

    def init_best_params(self):
        self.best_params['avg_prec_micro'] = 0
        self.best_params['avg_prec_macro'] = 0
        self.best_params['prec_micro'] = 0
        self.best_params['prec_macro'] = 0
        self.best_params['rec_micro'] = 0
        self.best_params['rec_macro'] = 0
        self.best_params['f1_score'] = 0
        self.best_params['log_loss'] = 100

    def fit(self, train_regressors=None, train_targets=None, C=1, w2v_model=None, no_tag=False):
        random_state = np.random.RandomState(0)

        self.C = C

        if self.name == 'nbayes':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        if self.name != 'cnn':
            self.scaler.fit(train_regressors)
            train_regressors = self.scaler.transform(train_regressors)
            logger.info('trainning %s: length and shape-> %i and %i; target shape -> %i'
                        % (self.name, train_regressors.shape[0], train_regressors.shape[1], len(train_targets[0])))

        if self.name == 'nbayes':
            classifier = OneVsRestClassifier(MultinomialNB(alpha=C), n_jobs=self.n_workers)

        elif self.name == 'dtree':
            classifier = OneVsRestClassifier(DecisionTreeClassifier(max_depth=C, criterion='gini',
                                                                    class_weight='balanced'),
                                             n_jobs=self.n_workers)

        elif self.name == 'rforest':
            classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=C, criterion='gini',
                                                                    class_weight='balanced', oob_score=True, verbose=1),
                                             n_jobs=self.n_workers)

        elif self.name == 'logistic':
            classifier = OneVsRestClassifier(
                LogisticRegression(solver='lbfgs', C=C, max_iter=200, verbose=1, multi_class='multinomial',
                                   random_state=random_state), n_jobs=self.n_workers)
        elif self.name == 'knn':
            classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=C, n_jobs=self.n_workers),
                                             n_jobs=self.n_workers)

        elif self.name == 'svm':
            classifier = OneVsRestClassifier(LinearSVC(C=C, max_iter=200, loss='squared_hinge',
                                                       # class_weight='balanced',
                                                       verbose=1, random_state=random_state),
                                             n_jobs=self.n_workers)
            # classifier = OneVsRestClassifier(SVC(kernel='rbf', probability=False, decision_function_shape='ovo',
            #                                     class_weight='balanced', verbose=1),
            #                                 n_jobs=self.n_workers)

        elif self.name == 'mlp':
            classifier = MLPClassifier(solver='adam', activation='logistic', alpha=C, max_iter=200,
                                       learning_rate_init=0.001,
                                       early_stopping=True, verbose=True, hidden_layer_sizes=(128,),
                                       random_state=random_state)

        elif self.name == 'cnn':
            K.clear_session()
            n_classes = 11
            max_doc_length = 1500
            classifier = CNN1D(w2v_model, n_classes, max_doc_length, filters=128, kernel_size=5, l2_lambda=C,
                               drop_out=0.5, batch_size=96, num_epochs=50, is_tag=(not no_tag), limit=None)

        self.predictor = classifier.fit(train_regressors, train_targets)

    def predict(self, test_regressors, queries):
        self.queries = queries

        if self.name != 'cnn':
            test_regressors = self.scaler.transform(test_regressors)
            # test_regressors = MinMaxScaler().fit_transform(test_regressors)

        self.predictions = self.predictor.predict(test_regressors)

    def predict_proba(self, test_regressors, queries):
        self.queries = queries
        if self.name != 'cnn':
            test_regressors = self.scaler.transform(test_regressors)
            # test_regressors = MinMaxScaler().fit_transform(test_regressors)
        if self.name == 'svm':
            self.predictions = self.predictor.predict(test_regressors)
        else:
            self.predictions = self.predictor.predict_proba(test_regressors)

        if self.name == 'cnn':
            self.queries = self.predictor.queries

    def eval(self, test_targets, labels=None, calc_threshold=True):
        if labels is None:
            labels = range(len(test_targets[0]))

        # Compute precsion recall curve and average precision for each class
        n_classes = len(labels)
        prec, rec, avg_prec, threshold = compute_metrics(test_targets, self.predictions, n_classes)
        if calc_threshold:
            threshold = compute_threshold(prec, rec, threshold, n_classes)
        else:
            threshold['best'] = self.best_params['threshold']

        y_ts = np.sum(test_targets) / (test_targets.shape[0] * test_targets.shape[1])

        p_name = self.name + '_' + str(self.C)
        plot_metric(prec, rec, avg_prec, p_name=p_name, threshold=y_ts)

        mclass_loss = log_loss(test_targets, self.predictions)

        for i in range(n_classes):
            self.predictions[self.predictions[:, i] >= threshold['best'][i], i] = 1
            self.predictions[self.predictions[:, i] < threshold['best'][i], i] = 0

        prec_micro = precision_score(test_targets, self.predictions, average='micro')
        prec_macro = precision_score(test_targets, self.predictions, average='macro')
        rec_micro = recall_score(test_targets, self.predictions, average='micro')
        rec_macro = recall_score(test_targets, self.predictions, average='macro')
        f1 = f1_score(test_targets, self.predictions, average='micro')

        if prec_micro > self.best_params['prec_micro']:
            logger.info('precision (micro): %f' % (prec_micro))
            self.best_params['c'] = self.C
            if calc_threshold:
                self.best_params['threshold'] = threshold['best']
            self.best_params['avg_prec_micro'] = avg_prec['micro']
            self.best_params['avg_prec_macro'] = avg_prec['macro']
            self.best_params['prec_micro'] = prec_micro
            self.best_params['prec_macro'] = prec_macro
            self.best_params['rec_micro'] = rec_micro
            self.best_params['rec_macro'] = rec_macro
            self.best_params['f1_score'] = f1
            self.best_params['log_loss'] = mclass_loss
            self.best_predictor = deepcopy(self.predictor)
            self.best_results = self.predictions

    def print_results(self, results=None, name=None, output_dir=None):
        if name is None:
            name = self.name
        if results is None:
            results = self.best_results
        filename = name + '.res'
        if output_dir is not None:
            filename = os.path.join(output_dir, filename)

        np.savetxt(filename, results, delimiter=',', fmt='%.4f')

    def print_results2(self, results=None, name=None, output_dir=None, test_set=[]):
        if name is None:
            name = self.name
        if results is None:
            results = self.best_results

        if len(test_set) == 0:
            test_set = self.queries

        results = self.inv_binarize(results)
        filename = name + '.res'
        if output_dir is not None:
            filename = os.path.join(output_dir, filename)
        with open(filename, 'w', encoding='utf-8') as f:
            for i in range(len(test_set)):
                qid = test_set[i].replace('_', '\t')
                labels = sorted([LABEL_MAP[r] for r in results[i] if r in LABEL_MAP])
                pclass = ''.join(['[' + cl + ']' for cl in labels])
                print(qid + '\t' + pclass, file=f)
        f.close()

    def binarize(self, targets):
        return load_mlb().transform(targets)

    def inv_binarize(self, results):
        results = np.asarray(results, dtype=int)
        return load_mlb().inverse_transform(results)

    def numpify(self, regressor):
        return np.array(regressor)

    def save_classifier_model(self, classifier_file, best=True):
        if best:
            # self.predictor = deepcopy(self.best_predictor)
            self.predictor = self.best_predictor
        self.C = self.best_params['c']
        # (self.predictions, self.best_results, self.best_predictor) = (None, None, None)
        (self.predictions, self.best_results) = (None, None)

        if not classifier_file.endswith('.pkl'):
            classifier_file = classifier_file + '.pkl'
        if self.name == 'cnn':
            _ending = '.h5py'
            # _ending = '.h5'
            hd5_file = classifier_file.replace('.pkl', _ending)
            self.predictor.save(hd5_file)
            del self.predictor.model
        with open(classifier_file, 'wb') as pickle_file:
            pickle.dump(self, pickle_file, pickle.HIGHEST_PROTOCOL)
        pickle_file.close()
        # joblib.dump(self.best_params['classifier'], classifier_file)

    @staticmethod
    def load_classifier_model(classifier_file):
        mcl = None
        with open(classifier_file, 'rb') as pickle_file:
            sys.modules['classifier'] = classifier
            mcl = pickle.load(pickle_file)
        pickle_file.close()
        if mcl.name == 'cnn':
            _ending = '.h5py'
            # _ending = '.h5'
            hd5_file = classifier_file.replace('.pkl', _ending)
            mcl.predictor.load(hd5_file)
        return mcl
        # self.predictor = joblib.load(classifier_file)

    @classmethod
    def load(cls, fname, mmap=None):
        """
        Load a previously saved object from file (also see `save`).

        """
        logger.info("loading %s object from %s" % (cls.__name__, fname))

        obj = cls.load_classifier_model(fname)
        return obj

    def save(self, classifier_file, best=True):
        """
        Save the object to file (also see `load`).

        `classifier_file` is a string specifying the file name to
        save to.

        """

        self.save_classifier_model(classifier_file, best=best)


class UniprotClassifier(CoreClassifier):
    def __init__(self, name, map_file=None, probability=False, n_workers=8):
        super().__init__(name, map_file=map_file, probability=probability, n_workers=n_workers)

    def print_results(self, results=None, category_map_file=None, repository_map_file=None):
        filename = self.name + '_uniprot_classifier.res'

        if category_map_file is not None:
            category_map = self.load_category(category_map_file)

        if repository_map_file is not None:
            repository_map_file = self.load_repository(repository_map_file)

        if self.probability:
            filename += '.prob'
        f = open(filename, encoding='utf8', mode='w')
        if results is None:
            results = self.best_results

        for j in range(len(self.queries)):

            if repository_map_file is not None:
                desc = '|'.join(sorted(repository_map_file[self.queries[j]]))
            else:
                desc = self.name + '_' + str(self.best_params['c'])

            score = {}
            for l in range(len(results[j])):
                score[self.mlb.classes_[l]] = results[j][l]
            if self.probability:
                score = self.norm_score(score)

            rank = 1
            psim = None
            for ncat, sim in sorted(score.items(), key=lambda x: x[1], reverse=True):
                if sim > 0:
                    if psim is not None and psim != sim:
                        rank += 1
                        psim = sim
                    elif psim is None:
                        psim = sim
                    if category_map_file is not None:
                        ncat = category_map[ncat]
                    print(self.queries[j], '0', ncat, rank, sim, desc, file=f)
            if psim is None:
                print(self.queries[j], '0', 'NA', 'NA', 'NA', desc, file=f)
        f.close()

    def load_repository(self, repository_map_file):
        repository_map = {}
        with open(repository_map_file, encoding='utf8') as f:
            for line_no, line in enumerate(f):
                (pmid, repo) = line.strip().split(' ')
                if pmid not in repository_map:
                    repository_map[pmid] = [repo]
                else:
                    repository_map[pmid].append(repo)
        f.close()
        return repository_map

    def load_category(self, category_map_file):
        category_map = {}
        with open(category_map_file, encoding='utf8') as f:
            for line_no, line in enumerate(f):
                data = line.strip().split(' ')
                category_map[data[0]] = data[1]
        f.close()
        return category_map


class kNNClassifier(object):
    def __init__(self, k, map):
        self.k = k
        self.map = map
        self.train_regressors = None
        self.mlb = None

    def fit(self, train_regressors, train_targets):
        self.train_regressors = train_regressors
        self.mlb = train_targets
        return self

    def predict(self, test_regressors):
        results = []
        predictions = []
        print(self.train_regressors)
        for vec in test_regressors:
            predictions.append(self.train_regressors.most_similar([vec], topn=self.k))

        for i in range(len(predictions)):
            tp = defaultdict(lambda: 0)
            # for each 'most_similar' document predicted
            for ndoc in predictions[i]:
                # provide its classes as final prediction
                for c in self.map[ndoc[0]]:
                    tp[c] += ndoc[1]  # add the similarity score for each class

            l = []
            for j in self.mlb.classes:
                if j in tp:
                    l.append(tp[j])
                else:
                    l.append(0)
            results.append(l)
        return results
