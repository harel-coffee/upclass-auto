#!/usr/bin/env python
from __future__ import print_function

import multiprocessing
import pickle
from datetime import datetime
from optparse import OptionParser

import nltk
import os
from gensim import utils as gsutils
from upclass.uniprot.input.article import parse_pmc, parse_pubmed, parse_biomed, \
    read_plain, read_sentence, extract_tokens, extract_tags, \
    clear_sentence, load_pmc_pmid_map, match_sentence


# P29358||P68251  7890696 1433B_SHEEP     14-3-3 protein beta/alpha||14-3-3 protein beta/alpha, N-terminally processed||KCIP-1||Protein kinase C inhibitor protein 1      YWHAB   PTM/processing||Unclassified
def load_map(source_file):
    class_map = {}
    # source = '/data/user/teodoro/uniprot/annotation/train_data.tsv'
    # source = '/data/user/teodoro/pycharm-repo/uniprot/scripts/ceci_test_data.tsv'
    print('loading training map data')
    with open(source_file, encoding='utf-8') as f:
        for line in f:
            (accession, pmid, protein, full_name, gene, up_class) = line.strip().split('\t')
            if pmid not in class_map:
                class_map[pmid] = {}
            if protein not in class_map[pmid]:
                class_map[pmid][protein] = {}
            else:
                print(protein, 'should not be here!')

            class_map[pmid][protein]['protein'] = clear_sentence(protein).split('_', 1)[0]
            class_map[pmid][protein]['accession'] = set([clear_sentence(ac) for ac in accession.split('||')])
            class_map[pmid][protein]['pmid'] = pmid
            class_map[pmid][protein]['full_name'] = set([clear_sentence(fn) for fn in full_name.split('||')])
            class_map[pmid][protein]['gene'] = set([clear_sentence(g) for g in gene.split('||')])
            class_map[pmid][protein]['up_class'] = set(up_class.split('||'))
    f.close()
    print('done loading')
    return class_map


# P29358||P68251  7890696 1433B_SHEEP     14-3-3 protein beta/alpha||14-3-3 protein beta/alpha, N-terminally processed||KCIP-1||Protein kinase C inhibitor protein 1      YWHAB   PTM/processing||Unclassified
def load_map_v2(source_file):
    class_map = {}
    # source = '/data/user/teodoro/uniprot/annotation/train_data.tsv'
    # source = '/data/user/teodoro/pycharm-repo/uniprot/scripts/ceci_test_data.tsv'
    acc_rel = set()
    query_file = '/data/user/teodoro/pycharm-repo/uniprot/resources/abb_ac2pmid_noCat.txt'
    with open(query_file, encoding='utf-8') as f:
        for line in f:
            (acc, pmid) = line.strip().split('\t')
            acc_rel.add(acc)

    print('loading training map data')
    with open(source_file, encoding='utf-8') as f:
        for line in f:
            (accession, _, protein, full_name, gene, up_class) = line.strip().split('\t')

            for acc in accession.split('||'):
                if acc in acc_rel:
                    if acc not in class_map:
                        class_map[acc] = {}
                    else:
                        continue
                        print(protein, 'should not be here!')

                    class_map[acc]['protein'] = clear_sentence(protein).split('_', 1)[0]
                    class_map[acc]['full_name'] = set([clear_sentence(fn) for fn in full_name.split('||')])
                    class_map[acc]['gene'] = set([clear_sentence(g) for g in gene.split('||')])
                    class_map[acc]['up_class'] = set(up_class.split('||'))
    f.close()
    print('done loading')
    return class_map


def extract_tags_biomed(sent_dict, prot_info, query):
    pre_text = {}
    stats = {}
    stats_factors = ['INFN', 'INGEN', 'INACC', 'INPROT']
    for i in stats_factors:
        stats[i] = 0

    for doc_section_type, sentences in sent_dict.items():
        if (doc_section_type is not None and doc_section_type != ''
                and sentences is not None and len(sentences) > 0):
            (pmid, section_type) = doc_section_type.split('_')

            for accession in query[pmid]:
                doc_id = pmid + '_' + accession
                if accession in prot_info:
                    info = prot_info[accession]
                else:
                    info = {'protein': ''}

                line_number = 1
                for sentence in sentences:
                    if doc_id not in pre_text:
                        pre_text[doc_id] = {}
                    if section_type not in pre_text[doc_id]:
                        pre_text[doc_id][section_type] = {}
                    pre_text[doc_id][section_type][line_number] = match_sentence(sentence, info)

                    line_number += 1
    return pre_text, stats


def get_cores(n_workers):
    cores = int(n_workers)
    if cores is None or cores > multiprocessing.cpu_count():
        cores = multiprocessing.cpu_count() - 1
        print('using %i cores' % (cores))
    if cores < 1:
        cores = 1
    return cores


def parse_source(args):
    source_file, dest_file, source = args

    pre_text = {}
    fid = os.path.splitext(os.path.split(source_file)[1])[0]

    if source == 'pmc':
        pre_text = parse_pmc(source_file)
    elif source == 'biomed':
        pre_text = parse_biomed(source_file)
    else:
        pre_text = parse_pubmed(source_file)

    if fid is not None:
        dest_dir = os.path.split(source_file)[0]
        os.makedirs(dest_dir, exist_ok=True)

        if dest_file.endswith('xml'):
            save_xml(dest_file, fid, pre_text=pre_text)
        else:
            save_text(dest_file, pre_text=pre_text)

    return fid


def parse_plain(args):
    source_file, dest_file = args
    pre_text = read_plain(source_file)
    pre_text = extract_tokens(pre_text, sent_extract=True)
    save_text(dest_file, pre_text=pre_text)
    return os.path.split(dest_file)[1]


def parse_tokenizer(args):
    source_file = args
    pre_text = read_plain(source_file)
    pre_text = extract_tokens(pre_text, sent_extract=False)
    return pre_text


def parse_sentence(args):
    source_file, dest_file, tag_info, query = args
    pre_text = read_sentence(source_file)
    if query is not None:
        pre_text, stats = extract_tags_biomed(pre_text, tag_info, query)
    else:
        pre_text, stats = extract_tags(pre_text, tag_info)

    for k, v in pre_text.items():
        dest_file_prot = dest_file + '_' + k
        save_text(dest_file_prot, pre_text=v)

    return os.path.split(dest_file)[1], stats


def work_source(source, dest, type='biomed', xml=False, n_workers=4):
    count = 0
    start_time, offset_time = (datetime.now(), datetime.now())
    dext = 'txt'
    if xml:
        dext = 'xml'

    ext = 'json'
    if type == 'pmc' or type == 'pubmed':
        ext = 'xml'

    cores = get_cores(n_workers)
    pool = multiprocessing.Pool(cores)
    jobs = ((os.path.join(source, fn), os.path.join(dest, os.path.splitext(fn)[0] + '.' + dext), type)
            for fn in os.listdir(source) if fn.endswith(ext))

    for group in gsutils.chunkize(jobs, chunksize=4 * cores, maxsize=1):
        for fid in pool.imap(parse_source, group):  # chunksize=100 ):
            count += 1
            offset_time = print_progress(count, offset_time)

    total_time = datetime.now() - start_time
    print('### ' + str(count) + ' files processed in ' + str(total_time.seconds) + ' seconds ###')


def work_tokenizer(source, dest, ext='txt', n_workers=4):
    count = 0
    start_time, offset_time = (datetime.now(), datetime.now())

    text = ''

    cores = get_cores(n_workers)
    pool = multiprocessing.Pool(cores)
    jobs = ((os.path.join(source, fn)) for fn in os.listdir(source) if fn.endswith(ext))

    for group in gsutils.chunkize(jobs, chunksize=4 * cores, maxsize=1):
        for pre_text in pool.imap(parse_tokenizer, group):  # chunksize=100 ):
            count += 1
            for k, v in pre_text.items():
                text += ' ' + v
            count += 1
            offset_time = print_progress(count, offset_time)

    dest_file = os.path.join(dest, 'pmc.pickle')

    # Make a new Tokenizer
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    tokenizer.train(text)

    out = open(dest_file, mode='wb')
    pickle.dump(tokenizer, out)
    out.close()

    total_time = datetime.now() - start_time
    print('### ' + str(count) + ' files processed in ' + str(total_time.seconds) + ' seconds ###')


def get_pmid(file_id, pmc_pmid_map):
    if file_id in pmc_pmid_map:
        file_id = pmc_pmid_map[file_id]
    return file_id


def work_plain(source, dest, ext='txt', n_workers=4):
    count = 0
    start_time, offset_time = (datetime.now(), datetime.now())
    dext = 'txt'

    # pmc_pmid_map = load_pmc_pmid_map()
    pmc_pmid_map = {}

    cores = get_cores(n_workers)
    pool = multiprocessing.Pool(cores)
    jobs = ((os.path.join(source, fn), os.path.join(dest, get_pmid(os.path.splitext(fn)[0], pmc_pmid_map) + '.' + dext))
            for fn in os.listdir(source) if fn.endswith(ext))

    for group in gsutils.chunkize(jobs, chunksize=4 * cores, maxsize=1):
        for fid in pool.imap(parse_plain, group):  # chunksize=100 ):
            count += 1
            offset_time = print_progress(count, offset_time)

    total_time = datetime.now() - start_time
    print('### ' + str(count) + ' files processed in ' + str(total_time.seconds) + ' seconds ###')


def work_sentence(source, dest, type, class_map, ext='txt', n_workers=4):
    count = 0
    start_time, offset_time = (datetime.now(), datetime.now())
    dext = 'txt'
    stats = {}

    cores = get_cores(n_workers)
    pool = multiprocessing.Pool(cores)

    print('### loading jobs')
    # source file, dest file (until PMID), info for PMID
    if type == 'biomed':
        query_file = '/data/user/teodoro/pycharm-repo/uniprot/resources/abb_ac2pmid_noCat.txt'
        query = {}
        with open(query_file, encoding='utf-8') as f:
            for line in f:
                (acc, pmid) = line.strip().split('\t')
                if pmid not in query:
                    query[pmid] = set()
                query[pmid].add(acc)

        jobs = ((os.path.join(source, fn), os.path.join(dest, fn), class_map, query) for fn in os.listdir(source)
                if fn.endswith(ext))
    else:
        pmc_pmid_map = load_pmc_pmid_map()
        jobs = ((os.path.join(source, fn), os.path.join(dest, get_pmid(os.path.splitext(fn)[0], pmc_pmid_map)),
                 class_map[get_pmid(os.path.splitext(fn)[0], pmc_pmid_map)], None) for fn in os.listdir(source) if
                fn.endswith(ext))
    print('### loading jobs done')
    # process the corpus in smaller chunks of docs, because multiprocessing.Pool
    # is dumb and would load the entire input into RAM at once...
    for group in gsutils.chunkize(jobs, chunksize=4 * cores, maxsize=2):
        for fid, f_stats in pool.imap(parse_sentence, group):  # chunksize=100 ):
            count += 1
            stats[fid] = f_stats
            offset_time = print_progress(count, offset_time)

    total_time = datetime.now() - start_time
    print('### ' + str(count) + ' files processed in ' + str(total_time.seconds) + ' seconds ###')

    for i, v in stats.items():
        print(i, v)


def print_progress(count, offset_time):
    offset = 1000
    if count % offset == 0:
        offset_time = datetime.now() - offset_time
        print('### [' + str(count) + '] ' + str(offset) + ' files processed in ' + str(
            offset_time.seconds) + ' seconds ###')
        offset_time = datetime.now()
    return offset_time


def save_text(output, pre_text):
    if pre_text is not None and len(pre_text) > 0:
        f = open(output, encoding='utf-8', mode='w')
        for k, v in pre_text.items():
            if isinstance(v, dict):
                for key, value in v.items():
                    if isinstance(value, list):
                        for value2 in value:
                            print(k + ' ' + str(key) + ' ' + value2, file=f)
                    elif isinstance(value, str):
                        print(k + ' ' + str(key) + ' ' + value, file=f)
                    else:
                        print('###[ERROR] unknown printing value type')
            elif isinstance(v, list):
                count = 0
                for value in v:
                    count += 1
                    print(k + ' ' + str(count) + ' ' + value, file=f)
            elif isinstance(v, str):
                print(k + ' ' + v, file=f)
            else:
                print('###[ERROR] unknown printing value type')
        f.close()


def save_xml(output, fileid, pre_text):
    if pre_text is not None and len(pre_text) > 0:
        f = open(output, encoding='utf-8', mode='w')

        print('<DOC>', file=f)
        print('<DOCNO>' + fileid + '</DOCNO>', file=f)
        if 'TITLE' in pre_text:
            print('<TITLE>' + pre_text['TITLE'] + '</TITLE>', file=f)
        if 'ABSTRACT' in pre_text:
            print('<TEXT>' + pre_text['ABSTRACT'] + '</TEXT>', file=f)
        print('</DOC>', file=f)

        f.close()


def main():
    usage = 'usage: %prog [options] arg1 arg2'
    parser = OptionParser(usage)
    parser.add_option('-s', '--source', dest='source',
                      help='read data from SOURCE DIR')
    parser.add_option('-d', '--destination', dest='destination',
                      help='store data to DEST DIR')
    parser.add_option('-x', '--xml', dest='xml', action='store_true',
                      help='save in xml format')
    parser.add_option('-t', '--type', dest='type',
                      help='source type [pmc, biomed, pubmed]')
    parser.add_option('-n', '--n_workers', dest='n_workers',
                      help='n workers')
    parser.add_option('-p', '--phase', dest='phase',
                      help='processing phase')
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose')
    parser.add_option('-q', '--quiet',
                      action='store_false', dest='verbose')

    (options, args) = parser.parse_args()

    if options.phase == '1':
        work_source(options.source, options.destination, options.type, xml=options.xml, n_workers=options.n_workers)
    elif options.phase == '1.5':
        work_tokenizer(options.source, options.destination, 'txt', n_workers=options.n_workers)
    elif options.phase == '2':
        # load_stops()
        # load_sent_tokeniyzer()
        work_plain(options.source, options.destination, 'txt', n_workers=options.n_workers)
    elif options.phase == '3':
        # source_file = '/data/user/teodoro/uniprot/annotation/train_data.tsv'
        source_file = '/data/user/teodoro/uniprot/annotation/NAR_annotation/acc_info.sprot'
        if options.type == 'biomed':
            class_map = load_map_v2(source_file)
        else:
            class_map = load_map(source_file)
        work_sentence(options.source, options.destination, options.type, class_map, ext='txt',
                      n_workers=options.n_workers)
    else:
        print('###[ERROR] unknown phase')


if __name__ == '__main__':
    main()
