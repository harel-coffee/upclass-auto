#!/usr/bin/env python
from __future__ import print_function

from urllib.request import urlopen

from lxml import etree

from upclass.uniprot.classifier.dataset import TaggedDataset, SentenceDataset
from upclass.uniprot.input.article import parse_pubmed, parse_pmc, extract_tokens, extract_tags, clear_sentence
from upclass.uniprot.input.regressors import get_feature_set
from upclass.uniprot.input.uniprot_entry import parse_from_accession

ENTREZ_BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=%s&id=%s&tool=%s&email=%s&format=xml'

TRANS_BASE = 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids=%s&tool=%s&email=%s'


def parse_from_url(url, db=None):
    pre_text = {}
    file = urlopen(url)
    if db == 'pmc':
        pre_text = parse_pmc(file)
    else:
        pre_text = parse_pubmed(file)

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


def get_article_dict(pmid, accession):
    article_dict = None
    pmid = str(pmid)
    fileid = pmid
    db = 'pubmed'
    try:
        pmcid = get_pmc_id(pmid)
        if pmcid is not None:
            fileid = pmcid.lower().replace('pmc', '')
            db = 'pmc'
            print('pmc', pmid, accession)

        article_dict = parse_from_id(fileid, db=db)

        if db == 'pmc' and ('INTRODUCTION' not in article_dict and (
                'ABSTRACT' not in article_dict or len(article_dict['ABSTRACT']) == 0)):
            fileid = pmid
            db = 'pubmed'
            article_dict = parse_from_id(fileid, db=db)
            print('-pmc', pmid, accession)
    except Exception as e:
        print('cannot process pmid', pmid)
        print(e)

    return article_dict


def get_tag_info(accession, accessions):
    tag_info = None
    if accessions is not None:
        for acc in accession:
            if acc in accessions:
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
    return tag_info, accessions


def tag_article(models, accession, pmid, articles=None, accessions=None, mtype='both'):
    pmid = str(pmid)
    article_dict = None
    if articles is None:
        articles = {}
    if pmid not in articles:
        article_dict = get_article_dict(pmid, accession[0])
        articles[pmid] = article_dict
    else:
        article_dict = articles[pmid]

    tag_info, accessions = get_tag_info(accession, accessions)
    feature_list = get_features(models, article_dict, tag_info, mtype)

    return feature_list, articles, accessions


def get_features(models, pmid, article_dict, tag_info, mtype):
    pre_text, stats = extract_tags(article_dict, tag_info)

    protein = list(tag_info.keys())[0]
    doc_id = str(pmid) + '_' + protein

    td = TaggedDataset()
    tag_doc = {}
    for doc in td.get_content_from_dict(doc_id, pre_text[protein]):
        tag_doc[doc.tags[0]] = doc.words

    no_tag_doc = {}
    if mtype != 'tag':
        td = SentenceDataset()
        for doc in td.get_content_from_dict(doc_id, article_dict):
            no_tag_doc[doc.tags[0]] = doc.words

    doc_tags = tag_doc.keys()
    doc_tags, feature_list = get_feature_set(models, doc_tags, [], [], text_tag=tag_doc, text_notag=no_tag_doc,
                                             mtype=mtype)

    return feature_list


def get_features_from_dict(pmid, article_dict, tag_info, mtype='tag'):
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
