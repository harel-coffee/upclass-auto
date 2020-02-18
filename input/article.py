#!/usr/bin/env python
from __future__ import print_function

import re
import socket
import json

import nltk
from bs4 import BeautifulSoup
from lxml import etree

timeout=120
socket.setdefaulttimeout(timeout)

abbreviation = ['a', 'å', 'Ǻ', 'Å', 'b', 'c', 'd', 'e', 'ɛ', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'Ö', 'Ø', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'µm',
                'abs', 'al', 'approx', 'bp', 'ca', 'cap', 'cf', 'co', 'd.p.c', 'dr', 'e.g', 'et', 'etc', 'er', 'eq',
                'fig', 'figs', 'h', 'i.e', 'it', 'inc', 'min', 'ml', 'mm', 'mol', 'ms', 'no', 'nt',
                'ref', 'r.p.m', 'sci', 's.d', 'sd', 'sec', 's.e.m', 'sp', 'ssp', 'st', 'supp', 'vs', 'wt']


def load_pmc_pmid_map():
    # ../annotation/pmc_pmid_map.tsv
    pmc_pmid_map = {}
    with open('/data/user/teodoro/uniprot/annotation/pmc_pmid_map.tsv', encoding='utf-8') as goa_f:
        for line_no, line in enumerate(goa_f):
            # PMCID_w PMID_wo
            ids = line.strip().split(' ', 1)
            if len(ids) == 2:
                pmid = ids[1].strip()
                pmcid = ids[0].strip()
                pmc_pmid_map[pmcid] = pmid
    return pmc_pmid_map


def parse_pmc(file):
    pre_text = {}

    # bulk parse
    parser = etree.XMLParser(encoding='utf-8', remove_blank_text=True)

    try:
        root = etree.parse(file, parser)
        pre_text = parse_title(root, pre_text)
        pre_text = parse_abstract(root, pre_text)

        pre_text, etrs = parse_caption(root, pre_text)

        for n in etrs:
            n.getparent().remove(n)

        pre_text = parse_body(root, pre_text)

    except Exception as e:
        print('error processing file', file)
        print('error', str(e))

    return pre_text


def parse_biomed(filename):
    # {result: [{MedlineCitation: {PMID:..., ArticleTitle: ..., Abstract: ...}}]}
    pre_text = {}

    # bulk parse
    try:
        f = open(filename, encoding='utf-8')
        mj = json.load(f)

        if 'result' in mj:
            for i in range(len(mj['result'])):
                try:
                    pmid = mj['result'][i]['MedlineCitation']['PMID']

                    if 'ArticleTitle' in mj['result'][i]['MedlineCitation']:
                        pre_text[pmid+'_TITLE'] = mj['result'][i]['MedlineCitation']['ArticleTitle']
                    if 'Abstract' in mj['result'][i]['MedlineCitation']:
                        pre_text[pmid+'_ABSTRACT'] = mj['result'][i]['MedlineCitation']['Abstract']
                except Exception as e:
                    print('error processing record '+ str(e))

        f.close()
    except Exception as e:
        print('error processing file', filename)
        print('error', str(e))

    return pre_text


def parse_pubmed(file):
    # <ArticleTitle>Evidence for a novel Cdc42GAP domain at the carboxyl terminus of BNIP-2. </ArticleTitle>
    # <Abstract><AbstractText>...</AbstractText></Abstract>
    pre_text = {}

    # bulk parse
    parser = etree.XMLParser(encoding='utf-8', remove_blank_text=True)

    try:
        root = etree.parse(file, parser)
        text = ''
        for element in root.iter('{*}ArticleTitle'):
            text += get_text_from_html(element) + ' '
        pre_text['TITLE'] = text

        text = ''
        for element in root.iter('{*}AbstractText'):
            text += get_text_from_html(element) + ' '
        pre_text['ABSTRACT'] = text

    except Exception as e:
        print('error processing file', file)
        print('error', str(e))

    return pre_text

phtml = re.compile('[\s\t]+')
def get_text_from_html(element):
    text = BeautifulSoup(etree.tostring(element), 'lxml').get_text(separator=u' ')
    return phtml.sub(' ', text.replace('\n', ' ').replace('\r', ''))


def parse_title(root, pre_text):
    text = ''
    # article-title
    for meta in root.iter('{*}article-meta'):
        for element in meta.iter('{*}article-title'):
            text += get_text_from_html(element) + ' '
    if text != '':
        pre_text['TITLE'] = text

    return pre_text


def parse_abstract(root, pre_text):
    text = ''
    # abstract
    for meta in root.iter('{*}article-meta'):
        for element in meta.iter('{*}abstract'):
            text += get_text_from_html(element) + ' '
    if text != '':
        pre_text['ABSTRACT'] = text
    return pre_text


def parse_body(root, pre_text):
    section = ''
    section_type = ''
    skip_section = ['web resources', 'reviewer\'s comments',
                    'pre-publication history', 'online data', 'list of abbreviations used', 'list of abbreviations',
                    'grants', 'funding', 'competing interests', 'availability of the software',
                    'availability and requirements',
                    'acknowledgments', 'abbreviations used', 'abbreviations',
                    'abbreviation', 'acknowledgements', 'availability of data and materials',
                    'availability of supporting data',
                    'competing interest', 'conflict of interest', 'conflict of interest statement', 'consent',
                    'data availability', 'database', 'database depositions', 'declaration of interest',
                    'disclosure', 'disclosures', 'extended data', 'for more information', 'fundings',
                    'note added in proof', 'notes', 'sources of funding']

    for body in root.iter('{*}body'):
        for element in body:
            if (element.tag == 'sec'):
                etitle = element.find('title')
                if etitle is not None and etitle.text is not None:
                    section = etitle.text.lower().strip()
                    etitle.getparent().remove(etitle)

            if section in skip_section or 'author' in section or 'supplement' in section:
                continue
            if section is None or section == '' or 'introduct' in section or 'background' in section or 'review' in section:
                section_type = 'INTRODUCTION'
            elif 'method' in section or 'material' in section or 'experiment' in section:
                section_type = 'METHODS'
            elif 'result' in section or 'finding' in section:
                section_type = 'RESULTS'
            elif 'discussion' in section or 'disscussion' in section:
                section_type = 'DISCUSSION'
            elif 'conclusion' in section or 'concluding' in section or 'summary' in section:
                section_type = 'CONCLUSION'
            else:
                section_type = 'UNKNOWN'
                print('###[WARNING] matching section: ' + section)

            text = get_text_from_html(element)
            if text != '':
                if section_type in pre_text:
                    pre_text[section_type] += ' ' + text
                else:
                    pre_text[section_type] = text
    return pre_text


def parse_caption(root, pre_text):
    fig, table, etr = ('', '', [])

    # fig
    for element in root.iter('{*}fig'):
        etr.append(element)
        for efig in element.iter('{*}caption'):
            fig += get_text_from_html(efig) + ' '

    # table
    for element in root.iter('{*}table-wrap'):
        etr.append(element)
        for etab in element.iter('{*}caption'):
            table += get_text_from_html(etab) + ' '

    # supplementary material
    for element in root.iter('{*}supplementary-material'):
        mat_type = ''
        etitle = element.find('label')
        if etitle is not None and etitle.text is not None:
            mat_type = etitle.text.strip().lower()

        if 'fig' in mat_type:
            etr.append(element)
            for etab in element.iter('{*}caption'):
                fig += get_text_from_html(etab) + ' '
        elif 'table' in mat_type:
            etr.append(element)
            for etab in element.iter('{*}caption'):
                table += get_text_from_html(etab) + ' '

    if fig != '':
        if 'FIGURE' in pre_text:
            pre_text['FIGURE'] = pre_text['FIGURE'] + ' ' + fig
        else:
            pre_text['FIGURE'] = fig

    if table != '':
        if 'TABLE' in pre_text:
            pre_text['TABLE'] = pre_text['TABLE'] + ' ' + table
        else:
            pre_text['TABLE'] = table

    return pre_text, etr


# 3. In Python, searching a set is much faster than searching
#   a list, so convert the stop words to a set
# stops = None
stops = set(nltk.corpus.stopwords.words('english'))

# def load_stops():
#    global stops
#    stops = set(nltk.corpus.stopwords.words('english'))


palfa = re.compile(r'[^\w\s]+')
nreal = re.compile(r'\b\d+\b')

stemmer = nltk.stem.PorterStemmer()


def clear_sentence(text):
    # 0. Work on lowercase
    text = text.lower()
    # 1. Remove useless characteres
    words = palfa.sub(' ', text)
    # 2. Convert to lower case, split into individual words
    words = words.split()
    # 3. Remove stop words and stem
    meaningful_words = [stemmer.stem(w) for w in words if w not in stops and len(w) > 1]
    # 4. Join the words back into one string separated by space,
    words = ' '.join(meaningful_words)
    # 5. Replace numbers
    words = nreal.sub('_NUMBER_', words)
    # Return the result
    return words


# sent_detector = None
sent_detector = nltk.data.load('/data/collection/douglas/tokenizer/pmc2.pickle')
sent_detector._params.abbrev_types.update(abbreviation)


# def load_sent_tokeniyzer():
#    global sent_detector
#    sent_detector = nltk.data.load('/data/collection/douglas/tokenizer/pmc2.pickle')
#    sent_detector._params.abbrev_types.update(abbreviation)


def extract_tokens(plain_dict, sent_extract=True, clean_sent=True):
    pre_text = {}
    # clean
    for section_type, section_text in plain_dict.items():
        sents = None
        if sent_extract:
            sents = sent_detector.tokenize(section_text.strip())
            if clean_sent:
                sents = [clear_sentence(sent) for sent in sents]
        else:
            sents = section_text.strip() + ' '
        if section_type in pre_text:
            pre_text[section_type] += sents
        else:
            pre_text[section_type] = sents
    return pre_text


def extract_tags(sent_dict, tag_info):
    pre_text = {}
    stats = {}
    stats_factors = ['INFN', 'INGEN', 'INACC', 'INPROT']
    for i in stats_factors:
        stats[i] = 0

    for section_type, sentences in sent_dict.items():
        if (section_type is not None and section_type != ''
                and sentences is not None and len(sentences) > 0):
            line_number = 1
            for sentence in sentences:
                #                if len(sentence.split()) <= 2 or len(sentence) < 20:
                #                    continue
                for protein, info in tag_info.items():
                    if protein not in pre_text:
                        pre_text[protein] = {}
                    if section_type not in pre_text[protein]:
                        pre_text[protein][section_type] = {}
                    pre_text[protein][section_type][line_number] = match_sentence(sentence, info)

                    for i in stats_factors:
                        if i in pre_text[protein][section_type][line_number]:
                            stats[i] += 1
                line_number += 1
    return pre_text, stats


def read_plain(filename):
    pre_text = {}
    f = open(filename, encoding='utf-8')
    for line in f:
        section = line.strip().split(' ', 1)
        if (len(section) == 2 and section[0] != '' and section[0] is not None and section[1] != '' and section[
            1] is not None):
            section_type = section[0]
            text = section[1]
            pre_text[section_type] = text
    f.close()
    return pre_text


def read_sentence(source_file):
    pre_text = {}

    f = open(source_file, encoding='utf-8')
    for line in f:
        section = line.strip().split(' ', 2)
        if section is not None and len(section) == 3 and section[2] is not None and len(section[2]) > 0:
            section_type, line_number, sentence = (section[0], section[1], section[2])
            line_number = int(line_number)
            if section_type not in pre_text:
                pre_text[section_type] = {}
            if line_number not in pre_text[section_type]:
                pre_text[section_type][line_number] = None
            pre_text[section_type][line_number] = sentence

    f.close()

    sort_text = {}
    for section in pre_text.keys():
        sort_text[section] = []
        for line_number in sorted(pre_text[section].keys()):
            sort_text[section].append(pre_text[section][line_number])

    return sort_text


def match_sentence(sentence, info):
    if len(info['protein']) > 2 and info['protein'] != '_NUMBER_':
        tgex = r'\b' + info['protein'] + r'\b'
        sentence = re.sub(tgex, ' _INPROT_ ', sentence)
    if 'full_name' in info:
        for v in sorted(info['full_name'], key=len, reverse=True):
            if v != '_NUMBER_' and len(v) > 2:
                tgex = r'\b' + re.escape(v) + r'\b'
                sentence = re.sub(tgex, ' _INFN_ ', sentence)
    if 'gene' in info:
        for v in sorted(info['gene'], key=len, reverse=True):
            if v != '_NUMBER_' and len(v) > 2:
                tgex = r'\b' + re.escape(v) + r'\b'
                sentence = re.sub(tgex, ' _INGEN_ ', sentence)
    if 'accession' in info:
        for v in sorted(info['accession'], key=len, reverse=True):
            if v != '_NUMBER_' and len(v) > 2:
                tgex = r'\b' + re.escape(v) + r'\b'
                sentence = re.sub(tgex, ' _INACC_ ', sentence)

    return sentence
