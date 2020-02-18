#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Douglas Teodoro <dhteodoro@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Model UniProt dataset
"""

from __future__ import print_function

from collections import namedtuple

import numpy as np
import os


def join_text(doc):
    new_doc = ' '.join(doc).split()
    new_doc = [t[:20] for t in new_doc if len(t) > 2]
    return new_doc


class TaggedDataset(object):
    def __init__(self, source_dir=None, only_var=False, neighbours=0, limit=None):
        # assert os.path.isdir(source_dir), 'dataset dir unavailable'
        self.source_dir = source_dir
        self.only_var = only_var
        self.neighbours = neighbours
        self.dataset = namedtuple('dataset', 'words tags')
        self.col_size = 0

        self.limit = limit

        self.IN = '_IN'
        self.OUT = '_OUT'
        self.IN_PROT = '_INPROT_'
        self.IN_FN = '_INFN_'
        self.IN_GEN = '_INGEN_'
        self.IN_ACC = '_INACC_'

    def __iter__(self):
        for d in self.get_dataset(self.get_content()):
            yield d

    def get_dataset(self, content):
        for doc_id, doc_in, doc_out in content:
            (did, protein) = doc_id.split('_', maxsplit=1)
            for i in range(2):
                if i == 0 and len(doc_in) > 0:
                    new_doc_id = did + '_IN_' + protein
                    self.col_size += 1
                    yield self.dataset(doc_in, [new_doc_id])
                elif i == 1 and len(doc_out) > 0:
                    new_doc_id = did + '_OUT_' + protein
                    self.col_size += 1
                    yield self.dataset(doc_out, [new_doc_id])

    def get_content(self, merged=True):
        count = 0
        for dirname, dirnames, filenames in os.walk(self.source_dir):
            for filename in filenames:
                source_file = os.path.join(dirname, filename)
                doc_id = filename
                if os.path.isfile(source_file):
                    sent_type = {}
                    sent_type[self.IN] = []
                    sent_type[self.OUT] = []
                    with open(source_file, encoding='utf8') as f:
                        for line_no, line in enumerate(f):
                            section, line_number, sent = line.strip().split(' ', 2)
                            line_number = str(line_number) + ' line' + str(round(int(line_number) / 5))
                            sent = section + ' ' + line_number + ' ' + sent
                            if self.IN in sent:
                                sent_type[self.IN].append(sent)
                            else:
                                sent_type[self.OUT].append(sent)
                    f.close()
                    if merged:
                        yield doc_id, join_text(sorted(sent_type[self.IN])), join_text(sorted(sent_type[self.OUT]))
                    else:
                        yield doc_id, sorted(sent_type[self.IN]), sorted(sent_type[self.OUT])

                    count += 1
                    if self.limit is not None and self.limit > 0 and count > self.limit:
                        break

    def get_content_from_dict(self, doc_id, tag_dict):
        sent_type = {}
        sent_type[self.IN] = []
        sent_type[self.OUT] = []
        for section_type in sorted(tag_dict.keys()):
            for line_number in sorted(tag_dict[section_type].keys()):
                sent = tag_dict[section_type][line_number]
                line_number = str(line_number) + ' line' + str(round(int(line_number) / 5))
                sent = section_type + ' ' + line_number + ' ' + sent
                if self.IN in sent:
                    sent_type[self.IN].append(sent)
                else:
                    sent_type[self.OUT].append(sent)
        ds_content = ((doc_id, join_text(sorted(sent_type[self.IN])), join_text(sorted(sent_type[self.OUT]))),)
        return self.get_dataset(ds_content)


class SentenceDataset(object):
    def __init__(self, source_dir=None, only_var=False, neighbours=0, category_map=None, limit=None):
        # assert os.path.isdir(source_dir), 'dataset dir unavailable'
        self.source_dir = source_dir
        self.dataset = namedtuple('dataset', 'words tags')
        self.col_size = 0
        self.logic_docs = None
        self.limit = limit

        if category_map is not None:
            self.logic_docs = {}
            for l_doc in category_map.keys():
                d_id, prot_id = l_doc.split('_', 1)
                if d_id not in self.logic_docs:
                    self.logic_docs[d_id] = []
                self.logic_docs[d_id].append(l_doc)

    def __iter__(self):
        for d in self.get_dataset(self.get_content()):
            yield d

    def get_dataset(self, content):
        for doc_id, sents in content:
            self.col_size += 1
            yield self.dataset(sents, [doc_id])

    def get_content(self):
        count = 0
        for dirname, dirnames, filenames in os.walk(self.source_dir):
            for filename in filenames:
                source_file = os.path.join(dirname, filename)
                doc_id = os.path.splitext(filename)[0]
                if os.path.isfile(source_file):
                    sents = []
                    with open(source_file, encoding='utf8') as f:
                        for line_no, line in enumerate(f):
                            info = line.strip().split(' ', 2)
                            if len(info) == 3:
                                section, line_number, sent = info
                                line_number = str(line_number) + ' line' + str(round(int(line_number) / 5))
                                sent = section + ' ' + line_number + ' ' + sent
                                sents.append(sent)
                    f.close()
                    if self.logic_docs is None or doc_id not in self.logic_docs:
                        yield doc_id, join_text(sorted(sents))
                    else:
                        lt = join_text(sorted(sents))
                        for logic_id in self.logic_docs[doc_id]:
                            yield logic_id, lt
                    count += 1
                    if self.limit is not None and self.limit > 0 and count > self.limit:
                        break

    def get_content_from_dict(self, doc_id, sent_dict):
        sent_type = []
        for section_type in sorted(sent_dict.keys()):
            line_number = 0
            for sent in sent_dict[section_type]:
                line_range = str(line_number) + ' line' + str(round(int(line_number) / 5))
                sent = section_type + ' ' + line_range + ' ' + sent
                sent_type.append(sent)
                line_number += 1
        ds_content = ((doc_id, sent_type),)
        return self.get_dataset(ds_content)


FEATURES = 'features'
LABELS = 'labels'
DOCS = 'docs'


class ProcessedDataset(object):
    def __init__(self, source_dir, type, eval=True):
        assert os.path.isdir(source_dir), 'dataset dir unavailable'
        self.source_dir = source_dir
        self.type = type
        self.eval = eval

    def get_content(self):
        # /data/user/teodoro/uniprot/dataset/no_large/processed/train_docs.csv
        labels = []
        doc_file = os.path.join(self.source_dir, self.type + '_' + DOCS + '.csv')
        if self.eval:
            label_file = os.path.join(self.source_dir, self.type + '_' + LABELS + '.csv')
            labels = self.get_csv_content(label_file)
        feature_file = os.path.join(self.source_dir, self.type + '_' + FEATURES + '.csv')
        docs = self.get_file_id(doc_file)
        features = self.get_csv_content(feature_file)
        # features = np.hstack((features[:,0:1], features[:,1:201]))
        # features = np.hstack((features, features[:, 401:601]))
        return docs, labels, features

    def get_file_id(self, cfile):
        content = []
        with open(cfile) as f:
            for line in f:
                content.append(line.strip())
        return content

    def get_csv_content(self, cfile):
        nfile = cfile.replace('.csv', '.npy')
        # if os.path.exists(nfile):
        #    content = np.load(nfile)
        # else:
        content = np.genfromtxt(cfile, delimiter=',')
        #    nfile = cfile.replace('.csv', '.npy')
        #    np.save(nfile, content)
        return content

    def get_npy_content(self, cfile):
        content = np.load(cfile)
        return content
