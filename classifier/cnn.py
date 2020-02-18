from __future__ import print_function

import time

import numpy as np
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dropout, BatchNormalization
from keras.layers import Embedding, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.layers import GlobalAveragePooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.models import load_model, save_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from numpy import zeros


from input.regressors import get_label_set

from copy import deepcopy
from keras.models import clone_model

if K.backend() == 'tensorflow':
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')
K.set_image_dim_ordering('th')


np.random.seed(0)


def doc_words(model):
    doc_index = {}
    doc_index['_PADDING_'] = 0
    for i, word in enumerate(model.wv.vocab):
        doc_index[word] = i + 1
    return doc_index


def encode_docs(dataset, word_index, logic_docs=None, limit=None):
    encoded_docs = []
    doc_tags = []
    max_doc_length = 0
    count_docs = 0
    word_count = {}
    for fid, tag_in in dataset:
        l_docvec, doc_wc = iterate_doc(tag_in, word_index)
        word_count, doc_length = update_word_count(doc_wc, word_count)

        if max_doc_length < doc_length:
            max_doc_length = doc_length
        if logic_docs is None or len(logic_docs) == 0:
            doc_tags.append(fid)
            encoded_docs.append(l_docvec)
        else:
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
    for fid, tag_in, tag_out in dataset:
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


def get_features(dataset, word_index, category_map=None, max_length=None, tag=True, limit=None):
    doc_encoded = []
    doc_tags = []
    max_doc_length = 0
    word_count = {}
    if tag:
        doc_encoded, doc_tags, max_doc_length, word_count = encode_doc_tag(dataset, word_index, limit=limit)
    else:
        logic_docs = {}
        if category_map is not None:
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


class CNN1D(object):
    def __init__(self, w2v_model, nclasses, max_doc_length, filters=128, kernel_size=5,
                 l2_lambda=0.0001, drop_out=0.5, num_epochs=50, batch_size=96, is_tag=True, limit=None):
        # Model Hyperparameters

        K.clear_session()

        self.nclasses = nclasses
        self.max_doc_length = max_doc_length

        # Training parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.l2_lambda = l2_lambda

        self.num_epochs = num_epochs
        self.drop_out = drop_out
        self.batch_size = batch_size

        self.is_tag = is_tag
        self.limit = limit

        self.name = 'cnn_notag'
        if self.is_tag:
            self.name = 'cnn_tag'

        self.model = None
        self.vocab_size = None
        self.embedding_dim = None
        self.word_index = None

        #        print(time.strftime('%c'), 'loading category map')
        #        self.category_map = get_class_map()
        embedding_matrix = self.init_model_features(w2v_model)
        if is_tag:
            self.cnn_model_tag(embedding_matrix)
        else:
            self.cnn_model(embedding_matrix)

        self.queries = None

        print('################################')
        print('## Model parameters')
        print('## nclasses', self.nclasses)
        print('## doc_length', self.max_doc_length)
        print('## filters', self.filters)
        print('## kernel_size', self.kernel_size)
        print('## num_epochs', self.num_epochs)
        print('## dropout_prob', self.drop_out)
        print('## vocab_size', self.vocab_size)
        print('## embedding_dim', self.embedding_dim)
        print('################################')

    def init_model_features(self, w2v_model):
        self.embedding_dim = w2v_model.vector_size
        print(time.strftime('%c'), 'loading word index')
        self.word_index = doc_words(w2v_model)
        self.vocab_size = len(self.word_index)
        print(time.strftime('%c'), 'loading embedding matrix')
        return embed_matrix(w2v_model, self.word_index, self.vocab_size, self.embedding_dim)

    def init_serial_model(self):

        input_shape = (self.doc_length, 1)
        model_input = Input(shape=input_shape, dtype='float32')

        conv = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(model_input)
        conv = MaxPooling1D(pool_size=self.kernel_size)(conv)
        conv = Dropout(self.dropout_prob[0])(conv)

        conv = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(conv)
        conv = MaxPooling1D(pool_size=self.kernel_size)(conv)
        conv = Dropout(self.dropout_prob[0])(conv)

        conv = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(conv)
        conv = GlobalAveragePooling1D()(conv)
        conv = Dropout(self.dropout_prob[0])(conv)

        z = Dense(self.filters, activation='relu')(conv)
        z = Dropout(self.dropout_prob[1])(z)
        model_output = Dense(self.nclasses, activation='softmax')(z)

        self.model = Model(inputs=model_input, outputs=model_output)

        # adam = optimizers.adam(lr=self.lr)
        adam = optimizers.adam()
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        print('total parameters:', self.model.count_params())
        plot_model(self.model, to_file='model.png')

    def fit(self, x_train, y_train, validation=None):
        print('model fitting - convolutional neural network')

        print('loading train features')
        x_train, train_docs, _, _ = get_features(x_train, self.word_index,
                                                 max_length=self.max_doc_length, tag=self.is_tag,
                                                 limit=self.limit)
        print('loading train labels')
        train_docs, y_train = get_label_set(train_docs, [], [], y_train)

        y_train = np.asarray(y_train)

        self.queries = train_docs
        # x_train = np.reshape(x_train, x_train.shape + (1,))
        print('x_train shape:', len(x_train[0]))
        print('y_train shape:', y_train.shape)

        print('number filter:', self.filters, 'filter size:', self.kernel_size)

        early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')

        #nn_weights = self.name + '_{}_.best.hdf5'.format(time.time())
        #checkpoint = ModelCheckpoint(nn_weights, save_weights_only=True, monitor='val_acc', verbose=2,
        #                             save_best_only=True,
        #                             mode='max')

        # tensor_board = TensorBoard(log_dir='/data/user/teodoro/tensorboard/graph_{}'.format(time.time()),
        #                            histogram_freq=2,
        #                            write_graph=True,
        #                            write_images=False,
        #                            embeddings_freq=2,
        #                            embeddings_layer_names=['embedding_1', 'embedding_2'])  # ,
        # embeddings_metadata={'embedding_1': METADATA_FILE})

        # fit the model
        print(time.strftime('%c'), 'training NN model')
        if validation is not None:
            print('loading dev features')
            x_dev = validation[0]
            y_dev = validation[1]
            x_dev, dev_docs, _, _ = get_features(x_dev, self.word_index,
                                                     max_length=self.max_doc_length, tag=self.is_tag,
                                                     limit=self.limit)
            print('loading dev labels')
            dev_docs, y_dev = get_label_set(dev_docs, [], [], y_dev)

            y_dev = np.asarray(y_dev)

            validation = (x_dev, y_dev)
        history = self.model.fit(x_train, y_train, validation_split=0.05,
                                    validation_data=validation,
                                    batch_size=96, shuffle=True,
                                    callbacks=[early_stopping], epochs=50, verbose=2)

        # K.clear_session()

        #self.model.load_weights(nn_weights)

        return self

    def predict(self, x_test):
        return np.rint(self.predict_proba(x_test))

    def predict_proba(self, x_test):
        print('model testing - convolutional neural network')
        print('x_test length:', len(x_test))

        x_test, test_docs, _, _ = get_features(x_test, self.word_index, max_length=self.max_doc_length,
                                               tag=self.is_tag, limit=self.limit)
        self.queries = test_docs
        # x_test = np.reshape(x_test, x_test.shape + (1,))
        print('x_test shape:', len(x_test[0]))

        print('number filter:', self.filters, 'filter size:', self.kernel_size)

        return self.model.predict(x_test, verbose=2)

    def save(self, classifier_file):
        _ending = '.h5py'
        #_ending = '.h5'
        if not classifier_file.endswith(_ending):
            classifier_file = classifier_file + _ending
        #self.model.save(classifier_file)
        save_model(self.model, classifier_file)
        # self.model.save_weights(classifier_file)

    def load(self, classifier_file):
        _ending = '.h5py'
        #_ending = '.h5'
        if not classifier_file.endswith(_ending):
            classifier_file = classifier_file + _ending
        self.model = load_model(classifier_file)
        # self.init_serial_model()
        #self.model.load_weights(classifier_file)

    def cnn_model(self, embedding_matrix):
        print(time.strftime('%c'), 'creating embedding layer')
        embedding_layer = Embedding(self.vocab_size, self.embedding_dim, weights=[embedding_matrix],
                                    input_length=self.max_doc_length,
                                    trainable=False)

        # define model
        print(time.strftime('%c'), 'creating NN model')

        sequence_input = Input(shape=(self.max_doc_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(self.filters, self.kernel_size, activation='relu')(embedded_sequences)
        x = BatchNormalization()(x)
        x = MaxPooling1D(self.kernel_size)(x)
        x = Dropout(self.drop_out)(x)

        x = Conv1D(self.filters, self.kernel_size, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(self.kernel_size)(x)
        x = Dropout(self.drop_out)(x)

        # x = Conv1D(128, 5, activation='relu')(x)
        # x = MaxPooling1D(5)(x)
        # x = Dropout(0.5)(x)

        x = Conv1D(self.filters, self.kernel_size, activation='relu')(x)
        x = BatchNormalization()(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(self.drop_out)(x)

        x = Dense(self.filters, activation='relu', kernel_regularizer=regularizers.l2(self.l2_lambda))(x)
        x = Dropout(self.drop_out)(x)
        preds = Dense(self.nclasses, activation='softmax', kernel_regularizer=regularizers.l2(self.l2_lambda))(x)

        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    def cnn_model_tag(self, embedding_matrix):

        print(time.strftime('%c'), 'creating embedding layer')
        #embedding_tag = Embedding(self.vocab_size, self.embedding_dim, weights=[embedding_matrix],
        #                          input_length=self.max_doc_length, trainable=False)

        #embedding_notag = Embedding(self.vocab_size, self.embedding_dim, weights=[embedding_matrix],
        #                            input_length=self.max_doc_length, trainable=False)

        embedding = Embedding(self.vocab_size, self.embedding_dim, weights=[embedding_matrix],
                            input_length=self.max_doc_length, trainable=False)

        # define model
        print(time.strftime('%c'), 'creating NN model')
        #######################
        tag_input = Input(shape=(self.max_doc_length,), dtype='int32')
        tag_sequences = embedding(tag_input)
        xt = Conv1D(self.filters, self.kernel_size, activation='relu')(tag_sequences)
        xt = BatchNormalization()(xt)
        xt = MaxPooling1D(self.kernel_size)(xt)
        xt = Dropout(self.drop_out)(xt)

        xt = Conv1D(self.filters, self.kernel_size, activation='relu')(xt)
        xt = BatchNormalization()(xt)
        xt = MaxPooling1D(self.kernel_size)(xt)
        xt = Dropout(self.drop_out)(xt)

        xt = Conv1D(self.filters, self.kernel_size, activation='relu')(xt)
        xt = BatchNormalization()(xt)
        xt = GlobalMaxPooling1D()(xt)

        #######################
        notag_input = Input(shape=(self.max_doc_length,), dtype='int32')
        notag_sequences = embedding(notag_input)
        xnt = Conv1D(self.filters, self.kernel_size, activation='relu')(notag_sequences)
        xnt = BatchNormalization()(xnt)
        xnt = MaxPooling1D(self.kernel_size)(xnt)
        xnt = Dropout(self.drop_out)(xnt)

        xnt = Conv1D(self.filters, self.kernel_size, activation='relu')(xnt)
        xnt = BatchNormalization()(xnt)
        xnt = MaxPooling1D(self.kernel_size)(xnt)
        xnt = Dropout(self.drop_out)(xnt)

        xnt = Conv1D(self.filters, self.kernel_size, activation='relu')(xnt)
        xnt = BatchNormalization()(xnt)
        xnt = GlobalMaxPooling1D()(xnt)
        #######################
        x = concatenate([xt, xnt])
        x = Dropout(self.drop_out)(x)

        x = Dense(self.filters, activation='relu', kernel_regularizer=regularizers.l2(self.l2_lambda))(x)
        x = Dropout(self.drop_out)(x)
        preds = Dense(self.nclasses, activation='softmax', kernel_regularizer=regularizers.l2(self.l2_lambda))(x)

        self.model = Model((tag_input, notag_input), preds)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'model':
                new_model = clone_model(v)
                new_model.set_weights(v.get_weights())
                setattr(result, k, new_model)
            else:
                setattr(result, k, deepcopy(v, memodict))
        return result
