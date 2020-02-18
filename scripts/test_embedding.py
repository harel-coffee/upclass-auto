import numpy as np
import time
from classifier.dataset import SentenceDataset
from gensim.models.doc2vec import Doc2Vec
from input.regressors import get_label_set
from input.utils import get_class_map
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense
from keras.layers import Embedding, Dropout
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import zeros
from sklearn.metrics import precision_score


def doc_words(model):
    doc_index = {}
    doc_index['_UNKNOW_'] = 0
    for i, word in enumerate(model.wv.vocab):
        doc_index[word] = i + 1
    return doc_index


def encode_docs(dataset, word_index, logic_docs):
    encoded_docs = []
    doc_tags = []
    max_doc_length = 0

    for doc in dataset:
        l_docvec = []
        count = 0
        for word in doc.words:
            if word in word_index:
                count += 1
                wi = word_index[word]
                l_docvec.append(wi)
        if max_doc_length < count:
            max_doc_length = count

        fid = doc.tags[0]
        for l_doc in logic_docs[fid]:
            doc_tags.append(l_doc)
            encoded_docs.append(l_docvec)
    return encoded_docs, doc_tags, max_doc_length


def embed_matrix(model, word_index, vocab_size, size):
    embedding_matrix = zeros((vocab_size, size), dtype='float32')
    for word, wi in word_index.items():
        if word in model.wv.vocab:
            word_vector = np.asarray(model.wv[word], dtype='float32')
            embedding_matrix[wi] = word_vector
    return embedding_matrix


def get_features(dataset, word_index, category_map, max_length=None):
    logic_docs = {}
    for l_doc in category_map.keys():
        d_id, prot_id = l_doc.split('_', 1)
        if d_id not in logic_docs:
            logic_docs[d_id] = []
        logic_docs[d_id].append(l_doc)

    doc_encoded, doc_tags, max_doc_length = encode_docs(dataset, word_index, logic_docs)

    # pad documents to a max length of 'max_length' words
    if max_length is None:
        max_length = max_doc_length

    doc_features = pad_sequences(doc_encoded, maxlen=max_length, padding='post', truncating='post')
    # print(train_docs)
    return doc_features, doc_tags, max_length


def cnn_model(vocab_size, embedding_dim, embedding_matrix, max_doc_length):
    print(time.strftime('%c'), 'creating embedding layer')
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_doc_length,
                                trainable=False)

    # define model
    print(time.strftime('%c'), 'creating NN model')

    sequence_input = Input(shape=(max_doc_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 10, activation='relu')(embedded_sequences)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(n_labels, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model


max_doc_length = 1500
n_labels = 11

print(time.strftime('%c'), 'loading doc2vec model')
model_file = '/data/user/teodoro/uniprot/model/no_tag/dbow'
model = Doc2Vec.load(model_file)

EMBEDDING_DIM = model.vector_size

print(time.strftime('%c'), 'loading word index')
word_index = doc_words(model)

vocab_size = len(word_index)

print(time.strftime('%c'), 'loading embedding matrix')
embedding_matrix = embed_matrix(model, word_index, vocab_size, EMBEDDING_DIM)

print(time.strftime('%c'), 'loading category map')
category_map = get_class_map()

# test data
source = '/data/user/teodoro/uniprot/dataset/no_large/test/sentence'
test_set = SentenceDataset(source)
print(time.strftime('%c'), 'loading test features')
test_features, test_docs, max_doc_length = get_features(test_set, word_index, category_map, max_length=max_doc_length)
print(time.strftime('%c'), 'loading test labels')
test_docs, test_labels = get_label_set(test_docs, [], [], category_map)
test_labels = np.asarray(test_labels)

# define model
nn_weights = 'cnn_weights.best.hdf5'
model = cnn_model(vocab_size, EMBEDDING_DIM, embedding_matrix, max_doc_length)

print('loading model from disk')
model.load_weights(nn_weights)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print('predicting')

y_pred = model.predict(test_features, verbose=1)

threshold = {0: 0.5,
             1: 0.5,
             2: 0.5,
             3: 0.5,
             4: 0.5,
             5: 0.5,
             6: 0.5,
             7: 0.5,
             8: 0.5,
             9: 0.5,
             10: 0.5}

for i in range(n_labels):
    y_pred[y_pred[:, i] >= threshold[i], i] = 1
    y_pred[y_pred[:, i] < threshold[i], i] = 0

prec_micro = precision_score(test_labels, y_pred, average='micro')

print('prec micro', prec_micro)

loss, accuracy = model.evaluate(test_features, test_labels, verbose=1)

print('loss', loss)
print('accuracy', accuracy)
