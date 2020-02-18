from __future__ import print_function

from collections import namedtuple

import os


def join_text(doc):
    new_doc = ' '.join(doc).split()
    new_doc = [t[:20] for t in new_doc if len(t) > 2]
    return new_doc


class BiomedDataset(object):
    def __init__(self, source_dir=None, category_map=None, limit=None):
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
                fid = os.path.splitext(filename)[0]
                if os.path.isfile(source_file):
                    docs = {}
                    with open(source_file, encoding='utf8') as f:
                        for line_no, line in enumerate(f):
                            info = line.strip().split(' ', 2)
                            if len(info) == 3:
                                id_section, line_number, sent = info
                                doc_id, section = id_section.split('_')
                                if doc_id not in docs:
                                    docs[doc_id] = []
                                line_number = str(line_number) + ' line' + str(round(int(line_number) / 5))
                                sent = section + ' ' + line_number + ' ' + sent
                                docs[doc_id].append(sent)
                    f.close()
                    if self.logic_docs is None or doc_id not in self.logic_docs:
                        for doc_id, sents in docs.items():
                            yield doc_id, join_text(sorted(sents))
                    else:
                        for doc_id, sents in docs.items():
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
