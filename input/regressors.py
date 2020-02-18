from random import shuffle
import datetime
import re
import numpy as np
from sklearn import preprocessing
import multiprocessing as mp

# ALPHA = 0.0001
# STEPS = 100
ALPHA = 0.025
MIN_ALPHA = 0.0001
STEPS = 100

def get_label_encoder():
    labels = ['ARATH', 'BACSU', 'BOVIN', 'CAEEL', 'CANAL', 'CHICK', 'DANRE', 'DICDI', 'DROME',
              'ECOLI', 'HUMAN', 'MOUSE', 'MYCTU', 'ORYSJ', 'OTHER', 'PIG', 'RAT', 'SCHPO',
              'XENLA', 'YEAST']
    lenc = preprocessing.LabelEncoder()
    lenc.fit(labels)
    # print(lenc.classes_)
    return lenc


def get_dev_test_pmids():
    source_dir = '/data/user/teodoro/uniprot/dataset/no_large/size_dist'
    colls = ['dev', 'test']
    sizes = ['small', 'large']
    dev_pmids, test_pmids = ([], [])
    for coll in colls:
        for size in sizes:
            with open(source_dir + '/' + size + '_' + coll) as f:
                for line in f:
                    if coll == 'dev':
                        dev_pmids.append(line.strip())
                    elif coll == 'test':
                        test_pmids.append(line.strip())
                    else:
                        print('this should not be here')
                        break
    return dev_pmids, test_pmids


def get_tag_comp(tag):
    if '_IN_' in tag or '_OUT_' in tag:
        info = tag.split('_', 2)
    else:
        info = tag.split('_', 1)
    return info


def get_org_code(tag):
    lenc = get_label_encoder()
    info = tag.split('_')
    corg = info[-1]
    if corg not in lenc.classes_:
        corg = 'OTHER'
    return lenc.transform([corg])[0]


def get_file_id(tag):
    return get_tag_comp(tag)[0]


def get_protein_tag(tag):
    return get_tag_comp(tag)[-1]


def get_label_set(doc_tags, in_set, out_set, cmap):
    doc_set, p_doc_tags, label_list = (set(), [], [])

    for i in doc_tags:
        fid = get_file_id(i)
        prot_tag = get_protein_tag(i)
        doc_tag = fid + '_' + prot_tag
        if (fid in in_set or (len(in_set) == 0 and fid not in out_set)) and doc_tag not in doc_set:
            doc_set.add(doc_tag)
            p_doc_tags.append(doc_tag)
            label_list.append(cmap[doc_tag])
    return p_doc_tags, label_list


def get_model_regressor(model, doc_tag_in, doc_tag_out):
    reg_in, reg_out = (None, None)

    if doc_tag_in in model.docvecs.doctags:
        reg_in = model.docvecs[doc_tag_in]
    if doc_tag_out in model.docvecs.doctags:
        reg_out = model.docvecs[doc_tag_out]
    return reg_in, reg_out


def infer_model_tagged_regressor(model, doc_tag_in, doc_tag_out, text_dataset):
    reg_in, reg_out = (None, None)

    if doc_tag_in in text_dataset and len(text_dataset[doc_tag_in]) > 0:
        # print('infering doc_tag_in')
        # print(text_dataset[doc_tag_in][:5])
        reg_in = model.infer_vector(doc_words=text_dataset[doc_tag_in], alpha=ALPHA, epochs=STEPS)
        # print(reg_in[:5])
    if doc_tag_out in text_dataset and len(text_dataset[doc_tag_out]) > 0:
        # print('infering doc_tag_out')
        # print(text_dataset[doc_tag_out][:5])
        reg_out = model.infer_vector(doc_words=text_dataset[doc_tag_out], alpha=ALPHA, epochs=STEPS)
        # print(reg_out[:5])
    return reg_in, reg_out


def get_tagged_regressor(model, fid, prot_tag, text_dataset=None):
    doc_tag_in = fid + '_IN_' + prot_tag
    doc_tag_out = fid + '_OUT_' + prot_tag
    reg_in, reg_out = (None, None)
    # print('text dataset', text_dataset)
    if text_dataset is not None:
        reg_in, reg_out = infer_model_tagged_regressor(model, doc_tag_in, doc_tag_out, text_dataset)
    else:
        reg_in, reg_out = get_model_regressor(model, doc_tag_in, doc_tag_out)
    if reg_in is None:
        reg_in = reg_out
    if reg_out is None:
        reg_out = np.zeros(len(reg_in))
    return np.concatenate([reg_in, reg_out])


def get_single_regressor(model, fid, text_dataset=None):
    reg = None
    if text_dataset is not None:
        if fid in text_dataset:
            reg = model.infer_vector(doc_words=text_dataset[fid], alpha=ALPHA, epochs=STEPS)
    elif fid in model.docvecs.doctags:
        reg = model.docvecs[fid]

    if reg is None:
        reg = np.zeros(model.vector_size)
    return reg


def __get_regressor(org_code, mtype, models, fid, prot_tag=None, text_tag=None, text_notag=None):
    lf = np.array([org_code])
    if mtype == 'tag' or mtype == 'both':
        if 'tag_dbow' in models:
            mreg = get_tagged_regressor(models['tag_dbow'], fid, prot_tag, text_dataset=text_tag)
            lf = np.concatenate([lf, mreg])
        if 'tag_dmc' in models:
            mreg = get_tagged_regressor(models['tag_dmc'], fid, prot_tag, text_dataset=text_tag)
            lf = np.concatenate([lf, mreg])
    if mtype == 'notag' or mtype == 'both':
        # if text_notag is not None and mtype != 'notag': # removed these two lines to work with inference no_tag
        #    fid = ndt                                  # there was a comment before saying they were inserted to be compatible with
        if 'notag_dbow' in models:  # text_url ???
            mreg = get_single_regressor(models['notag_dbow'], fid, text_dataset=text_notag)
            lf = np.concatenate([lf, mreg])
        if 'notag_dmc' in models:
            mreg = get_single_regressor(models['notag_dmc'], fid, text_dataset=text_notag)
            lf = np.concatenate([lf, mreg])
    return lf


def __get_job_params(doc_tags, in_set, out_set):
    doc_set = set()

    for odt in doc_tags:
        fid = get_file_id(odt)
        org_code = get_org_code(odt)
        prot_tag = get_protein_tag(odt)
        ndt = fid + '_' + prot_tag

        if (fid in in_set or (len(in_set) == 0 and fid not in out_set)) and ndt not in doc_set:
            doc_set.add(ndt)
            yield ndt, org_code, fid, prot_tag


def get_feature_set(models, doc_tags, in_set, out_set, text_tag=None, text_notag=None, mtype='tag', n_jobs=20):

    start_time = datetime.datetime.now()

    p_doc_tags, feature_list = [], []
    count = 0

    for ndt, org_code, fid, prot_tag in __get_job_params(doc_tags, in_set, out_set):
        reg = __get_regressor(org_code, mtype, models, fid, prot_tag, text_tag, text_notag)
        p_doc_tags.append(ndt)
        feature_list.append(reg)

        count += 1
        if count % 500 == 0:
            elapsed_time = datetime.datetime.now() - start_time
            print(count, 'docs processed in', elapsed_time.total_seconds(), 'sec')
            start_time = datetime.datetime.now()

    return p_doc_tags, feature_list


def infer_feature_set(models, dataset, in_set, out_set):
    doc_set, doc_tags, feature_list = (set(), [], [])

    for i in dataset.keys():
        fid = get_file_id(i)
        org_code = get_org_code(i)
        prot_tag = get_protein_tag(i)
        doc_tag = fid + '_' + prot_tag
        if (fid in in_set or (len(in_set) == 0 and fid not in out_set)) and doc_tag not in doc_set:
            doc_set.add(doc_tag)
            doc_tags.append(doc_tag)
            lf = np.array([org_code])
            for model in models:
                mreg = infer_model_tagged_regressor(model, fid, prot_tag, dataset)
                lf = np.concatenate([lf, mreg])
            feature_list.append(lf)
    return doc_tags, feature_list


def get_labelled_set(models, in_set, out_set, cmap, mtype='both'):
    doc_tags = None
    if 'tag_dbow' in models:
        doc_tags = models['tag_dbow'].docvecs.doctags
    elif 'tag_dmc' in models:
        doc_tags = models['tag_dmc'].docvecs.doctags
    elif 'notag_dbow' in models:
        doc_tags = models['notag_dbow'].docvecs.doctags
    elif 'notag_dmc' in models:
        doc_tags = models['notag_dmc'].docvecs.doctags
    p_doc_tags, label_list = get_label_set(doc_tags, in_set, out_set, cmap)
    p_doc_tags, feature_list = get_feature_set(models, doc_tags, in_set, out_set, mtype=mtype)

    return p_doc_tags, feature_list, label_list


def filter_single_doc(test_set, category_map):
    freq = []
    freq_set = {}
    for i in test_set:
        _doc_id = i[0]
        _doc = re.sub(r'_.*', '', _doc_id)
        _doc_class = sum([(i + 1) * j for i, j in enumerate(category_map[_doc_id])])
        if _doc not in freq_set:
            freq_set[_doc] = set()
        if _doc_class not in freq_set[_doc]:
            freq_set[_doc].add(_doc_class)
            freq.append(i)
    return freq


def filter_single_class(doc_tags, label_list, feature_list):
    label_count = {}
    good_docs = set()

    label_list = np.asarray(label_list, dtype=int)
    # convert class arrays to int
    s = ''
    for i in range(label_list.shape[1]):
        s += '{' + str(i) + '}'
    y_int = np.asarray([int(s.format(*label_list[i]), 2) for i in range(label_list.shape[0])])

    # select one document per class
    for i in range(len(doc_tags)):
        doc_tag = doc_tags[i]
        (doc_id, doc_suff) = doc_tag.split('_', 1)
        if doc_id not in label_count:
            label_count[doc_id] = {}
        doc_class = y_int[i]
        if doc_class not in label_count[doc_id]:
            label_count[doc_id][doc_class] = []
        label_count[doc_id][doc_class].append(doc_tag)
    for did, dt_values in label_count.items():
        for dc, dt_list in dt_values.items():
            shuffle(dt_list)
            good_docs.add(dt_list[0])

    # filter select documents
    new_doc_tags = []
    new_label_list = []
    new_feature_list = []
    for i in range(len(doc_tags)):
        doc_tag = doc_tags[i]
        if doc_tag in good_docs:
            new_doc_tags.append(doc_tag)
            new_label_list.append(label_list[i])
            new_feature_list.append(feature_list[i])
    return np.asarray(new_doc_tags), np.asarray(new_label_list), np.asarray(new_feature_list)
