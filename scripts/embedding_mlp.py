import numpy as np
import os
import time
from classifier.dataset import SentenceDataset, TaggedDataset
from classifier.multi_label_curve import compute_metrics, compute_threshold
from gensim.models.doc2vec import Doc2Vec
from input.regressors import get_label_set
from input.utils import get_class_map
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization, Dense
from keras.layers import Embedding, Dropout
from keras.layers import Input, Flatten
from keras.layers.merge import concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import zeros
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

MAX_DOC_LENGTH = 1500
N_LABELS = 11

# MODEL_FILE = '/data/user/teodoro/uniprot/model/no_tag/dbow'
MODEL_FILE = '/data/user/teodoro/uniprot/model/tag/dbow'

# train data
IS_CNN = True
IS_TAG = True
LIMIT = None

if IS_TAG:
    METADATA_FILE = '/data/user/teodoro/tensorboard/embedding_metadata_tag.tsv'
else:
    METADATA_FILE = '/data/user/teodoro/tensorboard/embedding_metadata_notag.tsv'


def doc_words(model):
    doc_index = {}
    doc_index['_PADDING_'] = 0
    for i, word in enumerate(model.wv.vocab):
        doc_index[word] = i + 1
    return doc_index


def encode_docs(dataset, word_index, logic_docs, limit=None):
    encoded_docs = []
    doc_tags = []
    max_doc_length = 0
    count_docs = 0
    word_count = {}
    for doc in dataset:
        l_docvec, doc_wc = iterate_doc(doc.words, word_index)
        word_count, doc_length = update_word_count(doc_wc, word_count)

        if max_doc_length < doc_length:
            max_doc_length = doc_length
        fid = doc.tags[0]
        for l_doc in logic_docs[fid]:
            doc_tags.append(l_doc)
            encoded_docs.append(l_docvec)
        count_docs += 1
        if limit is not None and count_docs >= limit:
            break

    return encoded_docs, doc_tags, max_doc_length, word_count


def update_word_count(doc_word_count, word_count):
    doc_count = 0
    for w, c in doc_word_count.items():
        if w not in word_count:
            word_count[w] = 0
        word_count[w] += c
        doc_count += c
    return word_count, doc_count


def encode_doc_tag(dataset, word_index, limit=None):
    encoded_tag = []
    encoded_notag = []
    doc_tags = []
    max_doc_length = 0
    count_docs = 0
    word_count = {}
    for fid, tag_in, tag_out in dataset.get_content():
        doc_tags.append(fid)

        tag_docvec, doc_wc = iterate_doc(tag_in, word_index)
        word_count, tag_length = update_word_count(doc_wc, word_count)

        notag_docvec, doc_wc = iterate_doc(tag_out, word_index)
        word_count, notag_length = update_word_count(doc_wc, word_count)

        encoded_tag.append(tag_docvec)
        encoded_notag.append(notag_docvec)
        if max_doc_length < max(tag_length, notag_length):
            max_doc_length = max(tag_length, notag_length)
        count_docs += 1
        if limit is not None and count_docs >= limit:
            break

    return [encoded_tag, encoded_notag], doc_tags, max_doc_length, word_count


def iterate_doc(doc, word_index):
    encoded_doc = []
    word_count = {}
    for word in doc:
        if word in word_index:
            wi = word_index[word]
            encoded_doc.append(wi)
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
    return encoded_doc, word_count


def embed_matrix(model, word_index, vocab_size, size):
    embedding_matrix = zeros((vocab_size, size), dtype='float32')
    for word, wi in word_index.items():
        if word in model.wv.vocab:
            word_vector = np.asarray(model.wv[word], dtype='float32')
            embedding_matrix[wi] = word_vector
    return embedding_matrix


def get_features(dataset, word_index, category_map, max_length=None, tag=True, limit=None):
    doc_encoded = []
    doc_tags = []
    max_doc_length = 0
    word_count = {}
    if tag:
        doc_encoded, doc_tags, max_doc_length, word_count = encode_doc_tag(dataset, word_index, limit=limit)
    else:
        logic_docs = {}
        for l_doc in category_map.keys():
            d_id, prot_id = l_doc.split('_', 1)
            if d_id not in logic_docs:
                logic_docs[d_id] = []
            logic_docs[d_id].append(l_doc)

        doc_encoded, doc_tags, max_doc_length, word_count = encode_docs(dataset, word_index, logic_docs, limit=limit)

    # pad documents to a max length of 'max_length' words
    if max_length is None:
        max_length = max_doc_length

    if tag:
        in_features = pad_sequences(doc_encoded[0], maxlen=max_length, padding='post', truncating='post')
        out_features = pad_sequences(doc_encoded[1], maxlen=max_length, padding='post', truncating='post')
        doc_features = [in_features, out_features]
    else:
        doc_features = pad_sequences(doc_encoded, maxlen=max_length, padding='post', truncating='post')
    # print(train_docs)
    return doc_features, doc_tags, max_doc_length, word_count


def mlp_model(vocab_size, embedding_dim, embedding_matrix, max_doc_length, n_labels):
    print(time.strftime('%c'), 'creating embedding layer')
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_doc_length,
                                trainable=False)

    # define model
    print(time.strftime('%c'), 'creating NN model')

    sequence_input = Input(shape=(max_doc_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Dense(128, activation='relu')(embedded_sequences)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    preds = Dense(n_labels, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model


def get_label_encoder():
    labels = ['ARATH', 'BACSU', 'BOVIN', 'CAEEL', 'CANAL', 'CHICK', 'DANRE', 'DICDI', 'DROME',
              'ECOLI', 'HUMAN', 'MOUSE', 'MYCTU', 'ORYSJ', 'OTHER', 'PIG', 'RAT', 'SCHPO',
              'XENLA', 'YEAST']
    ohenc = OneHotEncoder()
    ohenc.fit([[i] for i in range(len(labels))])
    # print(lenc.classes_)
    return ohenc


def encode_org(tag):
    lenc = get_label_encoder()
    info = tag.split('_')
    corg = info[-1]
    if corg not in lenc.classes_:
        corg = 'OTHER'
    return lenc.transform([corg])[0]


def org_input(length):
    _input = Input(shape=(length,), name='org_input', dtype='int32')
    return _input


def emb_input(vocab_size, embedding_dim, embedding_matrix, max_doc_length):
    print(time.strftime('%c'), 'creating embedding layer')
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_doc_length,
                                trainable=False)

    sequence_input = Input(shape=(max_doc_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    return embedded_sequences


def cnn_model(vocab_size, embedding_dim, embedding_matrix, max_doc_length, n_labels):
    print(time.strftime('%c'), 'creating embedding layer')
    l2_lambda = 0.0001
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_doc_length,
                                trainable=False)

    # define model
    print(time.strftime('%c'), 'creating NN model')

    sequence_input = Input(shape=(max_doc_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = BatchNormalization()(x)
    x = MaxPooling1D(5)(x)
    x = Dropout(0.5)(x)

    x = Conv1D(128, 5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(5)(x)
    x = Dropout(0.5)(x)

    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Dropout(0.5)(x)

    x = Conv1D(128, 5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = Dropout(0.5)(x)
    preds = Dense(n_labels, activation='softmax', kernel_regularizer=regularizers.l2(l2_lambda))(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model


def cnn_model_tag(vocab_size, embedding_dim, embedding_matrix, max_doc_length, n_labels):
    l2_lambda = 0.0001

    print(time.strftime('%c'), 'creating embedding layer')
    embedding_tag = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_doc_length,
                              trainable=False)

    embedding_notag = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_doc_length,
                                trainable=False)

    # define model
    print(time.strftime('%c'), 'creating NN model')

    tag_input = Input(shape=(max_doc_length,), dtype='int32')
    notag_input = Input(shape=(max_doc_length,), dtype='int32')

    tag_sequences = embedding_tag(tag_input)
    notag_sequences = embedding_notag(notag_input)

    xt = Conv1D(128, 10, activation='relu')(tag_sequences)
    xt = BatchNormalization()(xt)
    xt = MaxPooling1D(5)(xt)
    xt = Dropout(0.5)(xt)

    xt = Conv1D(128, 5, activation='relu')(xt)
    xt = BatchNormalization()(xt)
    xt = MaxPooling1D(5)(xt)
    xt = Dropout(0.5)(xt)

    xt = Conv1D(128, 5, activation='relu')(xt)
    xt = BatchNormalization()(xt)
    xt = GlobalMaxPooling1D()(xt)

    xnt = Conv1D(128, 10, activation='relu')(notag_sequences)
    xnt = BatchNormalization()(xnt)
    xnt = MaxPooling1D(5)(xnt)
    xnt = Dropout(0.5)(xnt)

    xnt = Conv1D(128, 5, activation='relu')(xnt)
    xnt = BatchNormalization()(xnt)
    xnt = MaxPooling1D(5)(xnt)
    xnt = Dropout(0.5)(xnt)

    xnt = Conv1D(128, 5, activation='relu')(xnt)
    xnt = BatchNormalization()(xnt)
    xnt = GlobalMaxPooling1D()(xnt)

    x = concatenate([xt, xnt])
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = Dropout(0.5)(x)
    preds = Dense(n_labels, activation='softmax', kernel_regularizer=regularizers.l2(l2_lambda))(x)

    model = Model((tag_input, notag_input), preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model


def fit_model(name, model, features, labels, val_features, val_labels):
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')

    nn_weights = name + '_{}_weights.best.hdf5'.format(time.time())
    checkpoint = ModelCheckpoint(nn_weights, save_weights_only=False, monitor='val_acc', verbose=2, save_best_only=True,
                                 mode='max')

    tensor_board = TensorBoard(log_dir='/data/user/teodoro/tensorboard/graph_{}'.format(time.time()),
                               histogram_freq=2,
                               write_graph=True,
                               write_images=False,
                               embeddings_freq=2,
                               embeddings_layer_names=['embedding_1', 'embedding_2'])  # ,
    # embeddings_metadata={'embedding_1': METADATA_FILE})

    # fit the model
    print(time.strftime('%c'), 'training NN model')
    model.fit(features, labels, validation_data=(val_features, val_labels),
              batch_size=96, shuffle=True,
              callbacks=[early_stopping, checkpoint, tensor_board], epochs=50, verbose=2)

    return model


print(time.strftime('%c'), 'loading doc2vec model')
# model_file = '/data/user/teodoro/uniprot/model/no_tag/dbow'
# model_file = '/data/user/teodoro/uniprot/model/tag/dbow'
model = Doc2Vec.load(MODEL_FILE)

EMBEDDING_DIM = model.vector_size

print(time.strftime('%c'), 'loading word index')
word_index = doc_words(model)

vocab_size = len(word_index)

print(time.strftime('%c'), 'loading embedding matrix')
embedding_matrix = embed_matrix(model, word_index, vocab_size, EMBEDDING_DIM)

print(time.strftime('%c'), 'loading category map')
category_map = get_class_map()

train_set = None
if IS_TAG:
    source = '/data/user/teodoro/uniprot/dataset/no_large/train/tag'
    train_set = TaggedDataset(source)
else:
    source = '/data/user/teodoro/uniprot/dataset/no_large/train/sentence'
    train_set = SentenceDataset(source)

print(time.strftime('%c'), 'loading train features')
train_features, train_docs, max_doc_length, word_count = get_features(train_set, word_index, category_map,
                                                                      max_length=MAX_DOC_LENGTH, tag=IS_TAG,
                                                                      limit=LIMIT)

if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        # print('Word\tFrequency', file=f)
        # print('_PADDING_\t0', file=f)
        print('_PADDING_', file=f)
        for word, index in sorted(word_index.items(), key=lambda x: x[1]):
            if word in word_count:
                # print(word+'\t'+str(word_count[word]), file=f)
                print(word, file=f)
    f.close()

print(time.strftime('%c'), 'max doc length', max_doc_length)
print(time.strftime('%c'), 'loading train labels')
train_docs, train_labels = get_label_set(train_docs, [], [], category_map)
train_labels = np.asarray(train_labels)

model = None
m_name = ''
if not IS_CNN:
    m_name = 'mlp'
    model = mlp_model(vocab_size, EMBEDDING_DIM, embedding_matrix, MAX_DOC_LENGTH, N_LABELS)
else:
    if IS_TAG:
        m_name = 'cnn_tag'
        # model = cnn_model(vocab_size, EMBEDDING_DIM, embedding_matrix, MAX_DOC_LENGTH, N_LABELS)
        model = cnn_model_tag(vocab_size, EMBEDDING_DIM, embedding_matrix, MAX_DOC_LENGTH, N_LABELS)
    else:
        m_name = 'cnn_notag'
        # model = cnn_model(vocab_size, EMBEDDING_DIM, embedding_matrix, MAX_DOC_LENGTH, N_LABELS)
        model = cnn_model(vocab_size, EMBEDDING_DIM, embedding_matrix, MAX_DOC_LENGTH, N_LABELS)

# summarize the model
print(time.strftime('%c'))
print(model.summary())

# dev data

dev_set = None
if IS_TAG:
    source = '/data/user/teodoro/uniprot/dataset/no_large/dev/tag'
    dev_set = TaggedDataset(source)
else:
    source = '/data/user/teodoro/uniprot/dataset/no_large/dev/sentence'
    dev_set = SentenceDataset(source)

print(time.strftime('%c'), 'loading dev features')
dev_features, dev_docs, max_doc_length, word_count = get_features(dev_set, word_index, category_map,
                                                                  max_length=MAX_DOC_LENGTH,
                                                                  tag=IS_TAG, limit=LIMIT)
print(time.strftime('%c'), 'loading dev labels')
dev_docs, dev_labels = get_label_set(dev_docs, [], [], category_map)
dev_labels = np.asarray(dev_labels, dtype=int)

model = fit_model(m_name, model, train_features, train_labels, dev_features, dev_labels)

##################################################
# evaluation metrics


print(time.strftime('%c'), 'evaluating NN model')
y_pred = model.predict(dev_features, verbose=2)

# # calculate best threshold
prec, rec, avg_prec, threshold = compute_metrics(dev_labels, y_pred, N_LABELS)
threshold = compute_threshold(prec, rec, threshold, N_LABELS)

for i in range(N_LABELS):
    y_pred[y_pred[:, i] >= threshold['best'][i], i] = 1
    y_pred[y_pred[:, i] < threshold['best'][i], i] = 0

prec_micro = precision_score(dev_labels, y_pred, average='micro')
fscore = f1_score(dev_labels, y_pred, average='micro')

print(time.strftime('%c'), 'threshold', threshold['best'])
print(time.strftime('%c'), 'dev prec micro', prec_micro)
print(time.strftime('%c'), 'dev fscore micro', fscore)

##################################################
# test data

test_set = None
if IS_TAG:
    source = '/data/user/teodoro/uniprot/dataset/no_large/test/tag'
    test_set = TaggedDataset(source)
else:
    source = '/data/user/teodoro/uniprot/dataset/no_large/test/sentence'
    test_set = SentenceDataset(source)

print(time.strftime('%c'), 'loading test features')
test_features, test_docs, max_doc_length, word_count = get_features(test_set, word_index, category_map,
                                                                    max_length=MAX_DOC_LENGTH,
                                                                    tag=IS_TAG, limit=LIMIT)
print(time.strftime('%c'), 'loading test labels')
test_docs, test_labels = get_label_set(test_docs, [], [], category_map)
test_labels = np.asarray(test_labels)

print('predicting test')
# loss, accuracy = model.evaluate(test_features, test_labels, verbose=2)

y_pred = model.predict(test_features, verbose=1)

for i in range(N_LABELS):
    y_pred[y_pred[:, i] >= threshold['best'][i], i] = 1
    y_pred[y_pred[:, i] < threshold['best'][i], i] = 0

prec_micro = precision_score(test_labels, y_pred, average='micro')
rec_micro = recall_score(test_labels, y_pred, average='micro')
fscore = f1_score(test_labels, y_pred, average='micro')
print(time.strftime('%c'), 'test prec micro', prec_micro)
print(time.strftime('%c'), 'test rec micro', rec_micro)
print(time.strftime('%c'), 'test fscore micro', fscore)
