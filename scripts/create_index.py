#!/usr/bin/env python
from __future__ import print_function

from optparse import OptionParser

from input.uniprot_entry import parse_from_file, print_record


def create_index_set(source_file, mapping=None, out_file=None):
    with open(out_file, mode='w') as f:
        for accession, doc_category, protein, protein_name, gene_name in parse_from_file(source_file, mapping=mapping):
            print_record(accession, doc_category, protein, protein_name, gene_name, out_file=f)
    f.close()


if __name__ == '__main__':
    usage = 'usage: %prog [options] arg1 arg2'
    parser = OptionParser(usage)
    parser.add_option('-d', '--dataset', dest='dataset', help='uniprot source file')
    parser.add_option('-m', '--mapping', dest='mapping', help='category mapping file')
    parser.add_option('-o', '--out_file', dest='out_file', help='result file')
    (options, args) = parser.parse_args()

    create_index_set(options.dataset, options.mapping, options.out_file)
