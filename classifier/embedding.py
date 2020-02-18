#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Douglas Teodoro <dhteodoro@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s -s FLAT_SOURCE_DIR -d OUTPUT_MODEL_DIR -l LANGUAGE [-n WORKERS] [-p]

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
import multiprocessing
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from random import shuffle
from timeit import default_timer

import plac
from gensim.models.doc2vec import Doc2Vec, FAST_VERSION

from collections import namedtuple

from classifier.dataset import TaggedDataset


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


class SentenceDataset(object):
    def __init__(self, source_dir=None, limit=None):
        # assert os.path.isdir(source_dir), 'dataset dir unavailable'
        self.source_dir = source_dir
        self.dataset = namedtuple('dataset', 'words tags')
        self.col_size = 0
        self.logic_docs = None
        self.limit = limit

    def __iter__(self):
        for d in self.get_content():
            yield d

    def get_content(self):
        count = 0
        for dirname, dirnames, filenames in os.walk(self.source_dir):
            for filename in filenames:
                source_file = os.path.join(dirname, filename)
                doc_id = os.path.splitext(filename)[0]
                if os.path.isfile(source_file):
                    words = ''
                    with open(source_file, encoding='utf8') as f:
                        for line_no, line in enumerate(f):
                            words += ' '+line.strip()
                    f.close()
                    if doc_id is not None:
                        yield self.dataset(words.split(), [doc_id])


def init_models(source, sentence=False, n_workers=1):
    assert FAST_VERSION > -1, 'this will be painfully slow otherwise'

    if n_workers is None:
        n_workers = multiprocessing.cpu_count() - 1

    # PV-DBOW
    models = [
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, vector_size=200, window=5, workers=n_workers,
                epochs=1, alpha=0.0250, min_alpha=0.0001, negative=5, hs=0, min_count=5),
        # PV-DM w/average
        # Doc2Vec(dm=1, dm_mean=1, size=200, window=5, workers=n_workers,
        #        iter=1, alpha=0.0250, min_alpha=0.0001, negative=5, hs=0, min_count=5),
        # PV-DBOW
        Doc2Vec(dm=0, dbow_words=1, vector_size=200, window=5, workers=n_workers,
                epochs=1, alpha=0.0250, min_alpha=0.0001, negative=5, hs=0, min_count=5),
    ]

    logger.info('loading corpus')
    train_docs = None
    with elapsed_timer() as elapsed:
        if sentence:
            alldocs = SentenceDataset(source)
        else:
            alldocs = TaggedDataset(source)
        # for i in alldocs:
        #    print(i)
        #    print('tags', i.tags)
        #    print('words', i.words)
        train_docs = [doc for doc in alldocs]

    logger.info('[%.1fs] end loading corpus' % (elapsed()))
    logger.info('corpus size: %i docs' % (len(train_docs)))

    # speed setup by sharing results of 1st model's vocabulary scan
    for i in range(len(models)):
        if i == 0:
            models[i].build_vocab(train_docs)
        else:
            models[i].reset_from(models[0])
        logger.info('built vocab for model %i %s' % (i, str(models[i])))

    models_by_name = OrderedDict()
    models_by_name['dbow'] = models[1]
    # models_by_name['dmm'] = models[1]
    models_by_name['dmc'] = models[0]

    return models_by_name, train_docs


def train_doc2vec(source, destination, sentence=False, n_workers=1):
    assert os.path.isfile(source) or os.path.isdir(source), 'dataset unavailable'
    assert os.path.isdir(destination), 'dest model dir unavailable'

    (models_by_name, train_docs) = init_models(source, sentence=sentence, n_workers=n_workers)

    logger.info('start training doc2vec %s' % datetime.now())

    for name, model in models_by_name.items():
        alpha, min_alpha, passes = (0.025, 0.0001, 100)
        alpha_delta = (alpha - min_alpha) / passes
        for epoch in range(passes):
            print('')
            logger.info('training model %s' % name)
            shuffle(train_docs)
            model.alpha, model.min_alpha = alpha, alpha
            with elapsed_timer() as elapsed:
                model.train(train_docs, total_examples=model.corpus_count, epochs=model.iter)
                logger.info('[%.1fs] epoch %i: end training model %s with alpha %.4f' % (elapsed(), epoch, name, alpha))
            alpha -= alpha_delta
        print('')
        logger.info('saving model %s' % name)
        if destination is not None:
            name = os.path.join(destination, name)
        model.save(name)

    logger.info('end training doc2vec %s\n' % str(datetime.now()))


@plac.annotations(
    source=('read data from source directory', 'option', 's', str),
    destination=('store model to destination directory', 'option', 'd', str),
    sentence=('use only sentence collection (no tag', 'flag', 'e'),
    n_workers=('number of parallel workers', 'option', 'n', int),

)
def main(source=None, destination=None, sentence=False, n_workers=1):
    train_doc2vec(source=source, destination=destination, sentence=sentence, n_workers=n_workers)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    plac.call(main)
