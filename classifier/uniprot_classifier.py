from __future__ import print_function

import os
import re
import logging
from pprint import pprint
from random import seed

import numpy as np

from classifier.model import CoreClassifier
from input.regressors import filter_single_class, filter_single_doc, get_label_set
from input.utils import order_test_set

seed(a=41)

random_state = np.random.RandomState(0)

logger = logging.getLogger('uniprot_classifier')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)


# def get_regressors_from_models(models=[], in_set=[], out_set=[], text_dataset=None):
#    doctags = models[0].docvecs.doctags
#    docids, regressors = get_feature_set(models, doctags, in_set, out_set, text_dataset=text_dataset)
#    return docids, regressors


# def get_labels_from_models(cmap, doctags, in_set=[], out_set=[]):
#    docids, labels = get_label_set(doctags, in_set, out_set, cmap)
#    return docids, labels


def train_uniprot_model(name, train_set, dev_set, output_dir, filter_train=True, w2v_model=None,
                        no_tag=False, category_map=None, n_workers=12):
    logger.info('training classifier %s' % (name))

    # prepare train and test input data
    train_targets, train_regressors, dev_targets, dev_regressors = (None, None, None, None)
    train_set = train_set.get_content()
    dev_set = dev_set.get_content()
    if name != 'cnn':
        if not filter_train:
            train_targets, train_regressors = (train_set[1], train_set[2])
            dev_targets, dev_regressors = (dev_set[1], dev_set[2])
        else:
            _, train_targets, train_regressors = filter_single_class(train_set[0], train_set[1], train_set[2])
            _, dev_targets, dev_regressors = filter_single_class(dev_set[0], dev_set[1], dev_set[2])
    else:
        # TODO implement CNN filter
        train_targets, train_regressors = (category_map, [i for i in train_set])
        dev_targets, dev_regressors = (category_map, [i for i in dev_set])
    logger.info('classes to learn: %i' % len(train_targets))

    # training parameter range
    if name == 'nbayes':
        # cs = [0.01, 0.1, 1, 10, 50, 100, 150, 200, 500]
        cs = np.logspace(-5, 1, 10)
    elif name == 'logistic':
        # cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1, 1.15]
        cs = np.logspace(-5, 1, 10)
    elif name == 'mlp':
        # cs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15]
        cs = np.logspace(-5, 1, 10)
    elif name == 'svm':
        # cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1, 1.15]
        cs = np.logspace(-5, 1, 10)
    elif name == 'dtree':
        cs = [1, 2, 5, 7, 10, 12, 15, 20]
    elif name == 'rforest' or name == 'knn':
        cs = [10, 25, 50, 100, 150, 200]
    elif name == 'cnn':
        #cs = [0.00001]
        cs = np.logspace(-5, 1, 10)  # l2_lambda
        # cs = [2, 3, 10, 25] #kernel size

    # train classifier
    best_classifier_model_file = os.path.join(output_dir, name)
    best_prec = 0
    count = 0
    for i in cs:

        classifier = CoreClassifier(name, n_workers=n_workers)

        logger.info('training classifier %s with param %f' % (classifier.name, i))
        classifier.fit(train_regressors, train_targets, C=i, w2v_model=w2v_model, no_tag=no_tag,
                       validation=(dev_regressors, dev_targets))

        print('len dev reg', len(dev_regressors))
        classifier.predict_proba(dev_regressors, dev_targets)
        print('len class queries', len(classifier.queries))
        if classifier.name == 'cnn':
            dev_docs, dev_labels = get_label_set(classifier.queries, [], [], dev_targets)
            dev_labels = np.asarray(dev_labels, dtype=int)
        else:
            dev_labels = dev_targets
        classifier.eval(dev_labels)
        if best_prec < classifier.best_params['prec_micro']:
            best_prec = classifier.best_params['prec_micro']
            classifier.save(best_classifier_model_file, best=False)
            count = 0
        else:
            count += 1

        if count >= 5:
            break
        print('')
        print('####################')
        print('####################')
    # print best parameters
    classifier = CoreClassifier(name, n_workers=n_workers)
    classifier = classifier.load_classifier_model(best_classifier_model_file)
    pprint(classifier.best_params)

    return classifier


def test_uniprot_model(classifier, test_set, filter=False, eval=False, query_doc=None, category_map=None):
    # load classifier
    logger.info('testing classifier %s' % (classifier.name))

    logger.info('classifier properties')
    logger.info(classifier.best_params)

    # test_docs, test_labels, test_features = train_set
    # prepare test data
    # eval data
    test_targets, test_regressors = (None, None)
    test_set = test_set.get_content()
    if query_doc is not None:
        test_set = order_test_set(test_set, query_doc)

    if classifier.name == 'cnn':
        logger.info('loading %s test data' % classifier.name)
        # TODO implement CNN filter
        if eval and filter:
            freq = filter_single_doc(test_set, category_map)
            test_targets, test_regressors = (category_map, freq)
        elif eval:
            test_targets, test_regressors = (category_map, [i for i in test_set])
        else:
            #test_targets, test_regressors = ([doc.tags[0] for doc in test_set], [i for i in test_set])
            test_regressors = [i for i in test_set]
            test_targets = [doc[0] for doc in test_regressors]
    else:
        if eval and filter:
            _, test_targets, test_regressors = filter_single_class(test_set[0], test_set[1], test_set[2])
        elif eval:
            test_targets, test_regressors = (test_set[1], test_set[2])
        # unseen data
        else:
            test_targets, test_regressors = (test_set[0], test_set[2])

    logger.info('%i queries to predict' % (len(test_targets)))

    logger.info('testing classifier %s with param %f' % (classifier.name, classifier.best_params['c']))
    logger.info(classifier.predictor)

    # test classifier
    classifier.predict_proba(test_regressors, test_targets)

    if eval:
        if classifier.name == 'cnn':
            test_docs, test_labels = get_label_set(classifier.queries, [], [], category_map)
            test_labels = np.asarray(test_labels)
        else:
            test_labels = test_targets

        classifier.init_best_params()
        classifier.eval(test_labels, calc_threshold=False)
        pprint(classifier.best_params)
    else:
        for i in range(len(classifier.best_params['threshold'])):
            classifier.predictions[classifier.predictions[:, i] >= classifier.best_params['threshold'][i], i] = 1
            classifier.predictions[classifier.predictions[:, i] < classifier.best_params['threshold'][i], i] = 0

    # print classifier results
    return classifier.predictions
