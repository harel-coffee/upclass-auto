import logging
import sys

import numpy as np
import os.path
import plac
from gensim.models.doc2vec import Doc2Vec
from random import seed

from upclass.uniprot.classifier.dataset import ProcessedDataset, TaggedDataset, SentenceDataset
from upclass.uniprot.classifier.model import CoreClassifier
from upclass.uniprot.classifier.uniprot_classifier import train_uniprot_model, test_uniprot_model
from upclass.uniprot.input.utils import get_class_map

seed(a=41)

random_state = np.random.RandomState(0)

MODEL_TAG_FILE = '/data/user/teodoro/uniprot/model/tag/dbow'
MODEL_NOTAG_FILE = '/data/user/teodoro/uniprot/model/no_tag/dbow'


@plac.annotations(
    classifier=('Classifier to use: nbayes, dtree, rforest, logistic, knn, mlp (default), svm, cnn', 'option', 'c', str,
                ['nbayes', 'dtree', 'rforest', 'logistic', 'knn', 'mlp', 'svm', 'cnn']),
    source_dir=('Location of the input train/test files', 'option', 's', str),

    output_dir=('Location of the model files', 'option', 'o', str),

    train_classifier=('Train classifier params (false default)', 'flag', 't'),

    classifier_model=('Trained classifier', 'option', 'm', str),

    filter_train=('Filter dupplicated train data', 'flag', 'f'),

    eval=('Eval results', 'flag', 'e'),

    query_doc=('Query document', 'option', 'q', str),

    no_tag=('No tag cnn', 'flag', 'u'),

    n_workers=('Number of workers', 'option', 'n', int),

)
def main(classifier='mlp', source_dir=None, output_dir=None, train_classifier=False, classifier_model=None,
         filter_train=False, eval=False, query_doc=None, no_tag=False, n_workers=None):
    assert classifier in ['nbayes', 'dtree', 'rforest', 'logistic', 'knn', 'mlp', 'svm', 'cnn'], \
        'classifier available: [nbayes, dtree, rforest, logistic, knn, mlp, svm, cnn]'

    # models = [Doc2Vec.load(vecmodel_dir + '/dbow'), Doc2Vec.load(vecmodel_dir + '/dmc')]

    category_map = None
    if train_classifier:
        w2v_model = None
        if classifier == 'cnn':
            category_map = get_class_map()
            if no_tag:
                w2v_model = Doc2Vec.load(MODEL_NOTAG_FILE)
                train_set = SentenceDataset(os.path.join(source_dir, 'train/sentence'), category_map=category_map,
                                            limit=None)
                dev_set = SentenceDataset(os.path.join(source_dir, 'dev/sentence'), category_map=category_map,
                                          limit=None)
            else:
                w2v_model = Doc2Vec.load(MODEL_TAG_FILE)
                train_set = TaggedDataset(os.path.join(source_dir, 'train/tag'))
                dev_set = TaggedDataset(os.path.join(source_dir, 'dev/tag'))
        else:
            train_set = ProcessedDataset(source_dir, type='train')
            dev_set = ProcessedDataset(source_dir, type='dev')

        model = train_uniprot_model(classifier, train_set, dev_set, w2v_model=w2v_model, filter_train=filter_train,
                                    no_tag=no_tag, category_map=category_map, n_workers=n_workers)
        classifier_file = os.path.join(output_dir, classifier + '_' + str(model.best_params['c']))
        model.save(classifier_file)

    else:
        if classifier == 'cnn':
            category_map = get_class_map()
            if no_tag:
                test_set = SentenceDataset(os.path.join(source_dir, 'test/sentence'), category_map=category_map,
                                           limit=None)
                # test_set = BiomedDataset(os.path.join(source_dir, 'test/sentence'), category_map=None, limit=None)
            else:
                test_set = TaggedDataset(os.path.join(source_dir, 'test/tag'))
                # test_set = TaggedDataset(source_dir)
        else:
            test_set = ProcessedDataset(source_dir, type='test', eval=eval)

        model = CoreClassifier(classifier)
        model = model.load(classifier_model)
        results = test_uniprot_model(model, test_set, filter=filter_train, eval=eval, query_doc=query_doc,
                                     category_map=category_map)

        if eval:
            model.print_results(results, name=classifier + '_' + str(model.best_params['c']),
                                output_dir=output_dir)
        else:
            model.print_results2(results, name=classifier + '_' + str(model.best_params['c']), output_dir=output_dir,
                                 test_set=[])


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    plac.call(main)
