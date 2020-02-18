#!/usr/bin/env python
from __future__ import print_function

import socket
from urllib.error import HTTPError
from urllib.request import urlopen

from lxml import etree

timeout=120
socket.setdefaulttimeout(timeout)

def get_mapping(source):
    category_map = {}
    with open(source, encoding='utf-8') as mapping:
        for line in mapping:
            codes = line.strip().split('|', maxsplit=2)
            category_map[codes[2]] = codes[0]
    mapping.close()
    return category_map


# <entry version="78" modified="2016-07-06" created="2004-10-25" dataset="Swiss-Prot">
#   <accession>P68251</accession>
#   <accession>P29358</accession>
#   <name>1433B_SHEEP</name>
#   <protein>
#     <recommendedName>
#       <fullName>14-3-3 protein beta/alpha</fullName>
#     </recommendedName>
#     <alternativeName>
#       <fullName>Protein kinase C inhibitor protein 1</fullName>
#       <shortName>KCIP-1</shortName>
#     </alternativeName>
#     <component>
#       <recommendedName>
#         <fullName>14-3-3 protein beta/alpha, N-terminally processed</fullName>
#       </recommendedName>
#     </component>
#   </protein>
#   <gene>
#     <name type="primary">YWHAB</name>
#   </gene>

# <reference key='3'>
#     <citation last='5709' first='5706' volume='270' name='J. Biol. Chem.' date='1995' type='journal article'>
#       <title>14-3-3 alpha and delta are the phosphorylated forms of raf-activating 14-3-3 beta and zeta. In vivo stoichiometric phosphorylation in brain at a Ser-Pro-Glu-Lys motif.</title>
#       <authorList>
#         <person name='Aitken A.'/>
#         <person name='Howell S.'/>
#         <person name='Jones D.'/>
#         <person name='Madrazo J.'/>
#         <person name='Patel Y.'/>
#       </authorList>
#       <dbReference id='7890696' type='PubMed'/>
#       <dbReference id='10.1074/jbc.270.11.5706' type='DOI'/>
#     </citation>
#     <scope>PHOSPHORYLATION AT SER-149</scope>
#     <scope>IDENTIFICATION BY MASS SPECTROMETRY</scope>
# </reference>

def parse_uniprot_entry(element):
    protein = None
    accession = []
    protein_name = []
    gene_name = []
    doc_category = {}

    protein = element.findtext('{*}name', default='')

    # print('protein', protein)
    # for name in element.iter('{*}name'):
    #    p_name = name.text.replace(',', '')
    # print(p_name)
    # if p_name is not None:
    #    p_name = p_name.text

    for scope in element.iter('{*}accession'):
        text = scope.text
        if text is not None:
            accession.append(text)

    # print('accession', accession)

    for scope in element.iter('{*}protein'):
        for eprot in scope.iter('{*}fullName'):
            text = eprot.text
            if text is not None:
                protein_name.append(text)
                # print('fullName', text)

        for eprot in scope.iter('{*}shortName'):
            text = eprot.text
            if text is not None:
                protein_name.append(text)
                # print('shortName', text)

    # print('protein_name', protein_name)

    for scope in element.iter('{*}gene'):
        for gene in scope.iter('{*}name'):
            text = gene.text
            if text is not None:
                gene_name.append(text)

    # print('gene_name', gene_name)

    for entry in element.iter('{*}reference'):
        pmid = None
        for etype in entry.iter('{*}dbReference'):
            if etype.get('type') and str(etype.get('type')).lower() == 'pubmed':
                pmid = etype.get('id')

        # print('pmid', pmid)

        if pmid is not None:
            doc_category[pmid] = []
            for scope in entry.iter('{*}scope'):
                text = scope.text
                # print('scope', text)
                if text is not None:
                    doc_category[pmid].append(text)

            # print('doc_category', doc_category)

    return accession, doc_category, protein, protein_name, gene_name


UNIPROT_BASE = 'http://www.uniprot.org/uniprot'


def parse_from_url(url):
    # category_map = get_mapping(mapping)
    # bulk parse
    # parser = etree.HTMLParser()
    parser = etree.XMLParser(encoding='utf-8', remove_blank_text=True)

    accession, pmid_scope, protein, protein_name, gene_name = (None, None, None, None, None)
    try:
        with urlopen(url, timeout=10) as f:
            tree = etree.parse(f, parser)
            for element in tree.iter('{*}entry'):
                accession, pmid_scope, protein, protein_name, gene_name = parse_uniprot_entry(element)
        f.close()
    except HTTPError:
        print('URL', url, 'could not be read.')

    return accession, pmid_scope, protein, protein_name, gene_name


def parse_from_accession(accession):
    url = UNIPROT_BASE + '/' + accession + '.xml'
    return parse_from_url(url)


def parse_from_file(filename, mapping=None):
    category_map = get_mapping(mapping)
    # bulk parse
    parser = etree.XMLParser(encoding='utf-8', remove_blank_text=True)

    root = etree.parse(filename, parser)

    for element in root.iter('{*}entry'):

        accession, pmid_scope, protein, protein_name, gene_name = parse_uniprot_entry(element)

        doc_category = {}
        for pmid, scope in pmid_scope.items():
            for text in scope:
                if text is not None and 'NUCLEOTIDE SEQUENCE [LARGE SCALE' not in text and text in category_map:
                    if mapping is not None and text in category_map:
                        cat = category_map[text]
                    else:
                        cat = text
                    if pmid not in doc_category:
                        doc_category[pmid] = []
                    if cat not in doc_category[pmid]:
                        doc_category[pmid].append(cat)

        yield accession, doc_category, protein, protein_name, gene_name


def print_record(accession, doc_category, protein, protein_name, gene_name, out_file=None):
    if len(doc_category) > 0:
        accession = '||'.join(sorted(accession))
        f_name = '||'.join(sorted(protein_name))
        g_name = '||'.join(sorted(gene_name))
        for pmid, categories in doc_category.items():
            a_class = '||'.join(sorted(categories))
            if out_file is None:
                print(accession + '\t' + pmid + '\t' + protein + '\t' + f_name + '\t' + g_name + '\t' + a_class)
            else:
                print(accession + '\t' + pmid + '\t' + protein + '\t' + f_name + '\t' + g_name + '\t' + a_class,
                      file=out_file)
