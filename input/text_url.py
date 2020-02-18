#!/usr/bin/env python
from __future__ import print_function

import os
import pickle
from random import shuffle
from urllib.request import urlopen

from gensim.models.doc2vec import Doc2Vec
from lxml import etree

from classifier.dataset import TaggedDataset, SentenceDataset
from input.article import parse_pubmed, parse_pmc, extract_tokens, extract_tags, clear_sentence
from input.regressors import get_feature_set
from input.uniprot_entry import parse_from_accession
from input.utils import load_mlb

ENTREZ_BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=%s&id=%s&tool=%s&email=%s&format=xml'

TRANS_BASE = 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids=%s&tool=%s&email=%s'


def parse_from_url(url, db=None):
    pre_text = {}
    with urlopen(url) as f:
        if db == 'pmc':
            pre_text = parse_pmc(f)
        else:
            pre_text = parse_pubmed(f)

    pre_text = extract_tokens(pre_text, sent_extract=True)

    return pre_text


def parse_from_id(fileid, db='pubmed'):
    fileid = str(fileid)
    tool = 'upclass'
    email = 'douglas.teodoro@hesge.ch'
    if fileid.lower().startswith('pmc'):
        db = 'pmc'
        fileid = fileid.lower().replace('pmc', '')

    url = ENTREZ_BASE % (db, fileid, tool, email)
    return parse_from_url(url, db=db)


def get_accession_data(accession):
    class_map = {}
    accession, pmid_scope, protein, protein_name, gene_name = parse_from_accession(accession)
    if protein not in class_map:
        class_map[protein] = {}
    class_map[protein]['protein'] = clear_sentence(protein)
    class_map[protein]['accession'] = [clear_sentence(ac) for ac in accession]
    class_map[protein]['full_name'] = [clear_sentence(fn) for fn in protein_name]
    class_map[protein]['gene'] = [clear_sentence(g) for g in gene_name]
    class_map[protein]['pmid'] = pmid_scope

    return class_map


def get_pmc_id(pmid):
    pmid = str(pmid)
    fileid = pmid
    tool = 'upclass'
    email = 'douglas.teodoro@hesge.ch'
    if fileid.lower().startswith('pmc'):
        db = 'pmc'
        fileid = fileid.lower().replace('pmc', '')

    url = TRANS_BASE % (fileid, tool, email)

    # bulk parse
    parser = etree.XMLParser(encoding='utf-8', remove_blank_text=True)

    pmcid = None
    with urlopen(url) as f:
        root = etree.parse(f, parser)
        for meta in root.iter('{*}record'):
            if meta.get('pmcid'):
                pmcid = meta.get('pmcid')

    return pmcid

def get_pmid_content(articles, pmid, accession):
    fileid = pmid
    db = 'pubmed'
    article_dict = None
    try:
        if articles is not None and pmid in articles:
            article_dict = articles[pmid]
        else:
            pmcid = get_pmc_id(pmid)
            if pmcid is not None:
                fileid = pmcid.lower().replace('pmc', '')
                db = 'pmc'
                print('pmc', pmid, accession[0])

            article_dict = parse_from_id(fileid, db=db)

            if db == 'pmc' and 'ABSTRACT' in article_dict and len(article_dict['ABSTRACT']) == 0:
                fileid = pmid
                db = 'pubmed'
                article_dict = parse_from_id(fileid, db=db)
                print('-pmc', pmid, accession[0])

            articles[pmid] = article_dict
    except Exception as e:
        print('cannot process pmid', pmid)
        print(e)

    return article_dict

def get_tag_info(accession, accessions):
    tag_info = None
    for acc in accession:
        if accessions is not None and acc in accessions:
            tag_info = accessions[acc]
            break

    if tag_info is None:
        for acc in accession:
            try:
                tag_info = get_accession_data(acc)
                accessions[acc] = tag_info
                break
            except Exception as e:
                print('cannot process accession', acc)
                print(str(e))
            finally:
                print('processed accession', acc)
    return tag_info


def get_feature_list(pmid, article_dict, tag_info, pre_text, models, mtype):
    protein = list(tag_info.keys())[0]
    doc_id = pmid + '_' + protein

    td = TaggedDataset()
    tag_doc = {}
    for doc in td.get_content_from_dict(doc_id, pre_text[protein]):
        tag_doc[doc.tags[0]] = doc.words

    td = SentenceDataset()
    no_tag_doc = {}
    for doc in td.get_content_from_dict(doc_id, article_dict):
        no_tag_doc[doc.tags[0]] = doc.words

    doc_tags = tag_doc.keys()
    doc_tags, feature_list = get_feature_set(models, doc_tags, [], [], text_tag=tag_doc, text_notag=no_tag_doc,
                                             mtype=mtype)
    return feature_list


def tag_article(accession, pmid, articles=None, accessions=None):
    pmid = str(pmid)

    article_dict = get_pmid_content(articles, pmid, accession)

    tag_info = get_tag_info(accession, accessions)

    pre_text, stats = extract_tags(article_dict, tag_info)

    # feature_list = get_feature_list(pmid, tag_info, pre_text, article_dict, models, mtype)

    return article_dict, tag_info, pre_text



# test_set = (('28229965', 'Q9LTJ1', ['Function', 'Subcellular location', 'Interaction']),
#             ('28167023', 'Q2LAJ3', ['Expression', 'Function']),
#             ('28157156', 'Q9CB01', ['Function', 'Pathology & Biotech', 'PTM/Processing']),
#             ('28150126', 'Q9C5U1', ['Pathology & Biotech']),
#             ('28150126', 'Q9C5U2', ['Pathology & Biotech']),
#             ('28120099', 'V9I1C0', ['Sequences']),
#             ('28084609', 'Q8LLE4', ['Expression']),
#             ('28084609', 'Q8LLD9', ['Expression', 'Pathology & Biotech']),
#             ('28084609', 'Q8LLE3', ['Expression', 'Pathology & Biotech']),
#             ('27988788', 'Q9XII1', ['Expression', 'Pathology & Biotech']),
#             ('27975189', 'Q9M126', ['Function', 'Pathology & Biotech', 'Expression']),
#             ('27832313', 'G7Z0A8', ['Sequences']),
#             ('27826761', 'Q9MA55', ['Function', 'Pathology & Biotech', 'Expression', 'Subcellular location']),
#             ('27826761', 'Q8RWD9', ['Function', 'Pathology & Biotech', 'Expression', 'Subcellular location']),
#             ('27770231', 'F4IAG1', ['Function', 'Pathology & Biotech', 'Subcellular location']),
#             ('27747895', 'O64733', ['Expression']),
#             ('27717956', 'B4FLA3', ['Sequences']),
#             ('27716933', 'Q9XER6', ['Expression']),
#             ('27704232', 'Q40133', ['Function', 'Pathology & Biotech']),
#             ('27681945', 'P42813', ['Function']),
#             ('27545692', 'D1GZJ3', ['Function']))

test_set = (('28229965', 'Q9LTJ1', ['Function', 'Subcellular location', 'Interaction']),
            ('28167023', 'Q2LAJ3', ['Expression', 'Function']),
            ('28157156', 'Q9CB01', ['Function', 'Pathology & Biotech', 'PTM/Processing', 'Expression']),
            ('28150126', 'Q9C5U1', ['Pathology & Biotech']),
            ('28150126', 'Q9C5U2', ['Pathology & Biotech']),
            ('28084609', 'Q8LLE4', ['Expression']),
            ('28084609', 'Q8LLD9', ['Expression', 'Pathology & Biotech']),
            ('28084609', 'Q8LLE3', ['Expression', 'Pathology & Biotech']),
            ('27988788', 'Q9XII1', ['Expression', 'Pathology & Biotech']),
            ('27975189', 'Q9M126', ['Function', 'Pathology & Biotech', 'Expression']),
            ('27832313', 'G7Z0A8', ['Sequence']),
            ('27826761', 'Q9MA55', ['Function', 'Pathology & Biotech', 'Expression', 'Subcellular location']),
            ('27826761', 'Q8RWD9', ['Function', 'Pathology & Biotech', 'Expression', 'Subcellular location']),
            ('27770231', 'F4IAG1', ['Function', 'Pathology & Biotech', 'Subcellular location']),
            ('27747895', 'O64733', ['Expression', 'Pathology & Biotech']),
            ('27717956', 'B4FLA3', ['Sequence']),
            ('27716933', 'Q9XER6', ['Expression', 'Function']),
            ('27704232', 'B1Q3F2', ['Function', 'Pathology & Biotech']),
            ('27681945', 'P42813', ['Function', 'Expression']),
            ('27545692', 'D1GZJ3', ['Function', 'Expression']))

import numpy as np

# test_source = 'ceci'
# test_source = 'new_sp'
# test_source = 'ceci_new'
test_source = 'pub'

# test_set = None
eval = True
# if test_source == 'pub_set':
#     test_set = '/data/user/teodoro/uniprot/annotation/new_annotation/test_set_pub.tsv'
#     eval = True
# elif test_source == 'ceci':
#     test_set = '/data/user/teodoro/uniprot/annotation/new_annotation/test_set_ceci.tsv'
#     eval = True
# elif test_source == 'ceci_new':
#     test_set = '/data/user/teodoro/uniprot/annotation/new_annotation/test_set_ceci_new.tsv'
#     eval = False
# else:
#     test_set = '/data/user/teodoro/uniprot/annotation/new_annotation/test_set.tsv'
#     eval = True

regressors = []
labels = []
docs = []
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

models_tag = '/data/user/teodoro/uniprot/model/pub_model/tag'
models_notag = '/data/user/teodoro/uniprot/model/pub_model/no_tag'
models = {
    'tag_dbow': Doc2Vec.load(models_tag + '/dbow'),
    # 'tag_dmc': Doc2Vec.load(models_tag + '/dmc'),
    # 'notag_dbow': Doc2Vec.load(models_notag + '/dbow'),
    'notag_dmc': Doc2Vec.load(models_notag + '/dmc')
}

count = 0
mtype = None
if len(models) == 4:
    mtype = 'both'
elif len(models) == 2:
    mtype = 'tag'
elif len(models) == 3:
    mtype = 'notag'
else:
    exit()

# for pmid, accession, target in test_set:
# with open(test_set) as test_file:
#     for line in test_file:
#         if eval:
#             (accession_all, pmid, target) = line.strip().split('\t')
#         else:
#             (accession_all, pmid) = line.strip().split('\t')
#         accession = accession_all.split('||')
#         shuffle(accession)
for (pmid, accession, target) in test_set:
    accession = [accession]
    print('processing %s %s' % (accession, pmid))
    try:
        article_dict, tag_info, pre_text = tag_article(accession, pmid, articles=articles, accessions=accessions)

        # feature_list = get_feature_list(pmid, article_dict, tag_info, pre_text, models, mtype)
        # regressors.append(feature_list[0])

        query_id = pmid + '_' + accession[0]
        docs.append(query_id)

        with open("/data/user/teodoro/pycharm-repo/uniprot/resources/tests/annotation/%s" % query_id, mode='w') as f:
            for p in pre_text.keys():
                for k, v in pre_text[p].items():
                    for line, sent in v.items():
                        print(query_id, k, line, sent, file=f)

        # if eval:
        #     mt = [t.lower() for t in target.split('||')]
        #     #            mt = [t.lower() for t in target]
        #     bin_labels = mlb.transform([mt])[0]
        #     labels.append(bin_labels)
    except Exception as e:
        print('could not process', pmid, accession)
        print(str(e))
    count += 1
    if count % 50 == 0:
        print(count, 'processed')

# if test_source != 'ceci':
#     if not os.path.exists(art_file):
#         with open(art_file, 'wb') as f:
#             # Pickle the 'data' dictionary using the highest protocol available.
#             pickle.dump(articles, f, pickle.HIGHEST_PROTOCOL)
#         f.close()
#
#     if not os.path.exists(acc_file):
#         with open(acc_file, 'wb') as f:
#             # Pickle the 'data' dictionary using the highest protocol available.
#             pickle.dump(accessions, f, pickle.HIGHEST_PROTOCOL)
#         f.close()
#
# fa = np.asarray(regressors)
# np.savetxt('/data/user/teodoro/uniprot/dataset/no_large/processed/' + test_source + '/' + mtype + '/test_features.csv',
#            fa, delimiter=',',
#            fmt='%.5f')
#
# if eval:
#     la = np.asarray(labels)
#     np.savetxt(
#         '/data/user/teodoro/uniprot/dataset/no_large/processed/' + test_source + '/' + mtype + '/test_labels.csv', la,
#         delimiter=',',
#         fmt='%i')
#
# with open('/data/user/teodoro/uniprot/dataset/no_large/processed/' + test_source + '/' + mtype + '/test_docs.csv',
#           mode='w') as f:
#     for i in docs:
#         print(i, file=f)
# f.close()
