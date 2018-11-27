from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.activations import softmax
import gensim
from gensim.models import Word2Vec

BASE_DIR = ''
GLOVE_DIR = ''
# TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 10000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

print('Indexing word vectors.')

# embeddings_index = {}
# with open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt')) as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs

# print('Found %s word vectors.' % len(embeddings_index))

print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
# for name in sorted(os.listdir(TEXT_DATA_DIR)):
#     path = os.path.join(TEXT_DATA_DIR, name)
#     if os.path.isdir(path):
#         label_id = len(labels_index)
#         labels_index[name] = label_id
#         for fname in sorted(os.listdir(path)):
#             if fname.isdigit():
#                 fpath = os.path.join(path, fname)
#                 args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
#                 with open(fpath, **args) as f:
#                     t = f.read()
#                     i = t.find('\n\n')  # skip header
#                     if 0 < i:
#                         t = t[i:]
#                     texts.append(t)
#                 labels.append(label_id)


# with open("Categorized_User_Polarity.txt") as inp:
#     for line in inp:
#         # print (line)
#         values    = line.split("\t") # id, polarity
#         user_id   = values[0]
#         user_file = "tokens_lines_test_2/" + user_id+".txt"
#         if os.path.isfile(user_file): # not all user IDs had tweets in the master tweet file
#             labels.append(int(values[1])) # save the polarity
#             with open(user_file, 'r') as inp: # save the
#                 texts.append(inp.read())


cls_0_num       = 0
cls_1_num       = 0
cls_2_num       = 0
cls_3_num       = 0
cls_4_num       = 0
cls_num_ext     = 600
with open("Categorized_User_Polarity.txt") as inp:
    for line in inp:
        values    = line.split("\t") # id, polarity
        user_id   = values[0]
        user_file = "tokens_lines_test_1/" + user_id+".txt"
        if os.path.isfile(user_file): # not all user IDs had tweets in the master tweet file
            if int(values[1]) == 0:
                if cls_0_num < cls_num_ext:
                    labels.append(int(values[1])) # save the polarity
                    with open(user_file, 'r') as inp: # save the
                        texts.append(inp.read())
                cls_0_num += 1
            if int(values[1]) == 0:
                if cls_1_num < cls_num_ext:
                    labels.append(int(values[1])) # save the polarity
                    with open(user_file, 'r') as inp: # save the
                        texts.append(inp.read())
                cls_1_num += 1
            if int(values[1]) == 0:
                if cls_2_num < cls_num_ext:
                    labels.append(int(values[1])) # save the polarity
                    with open(user_file, 'r') as inp: # save the
                        texts.append(inp.read())
                cls_2_num += 1
            if int(values[1]) == 0:
                if cls_3_num < cls_num_ext:
                    labels.append(int(values[1])) # save the polarity
                    with open(user_file, 'r') as inp: # save the
                        texts.append(inp.read())
                cls_3_num += 1
            else: # class 4
                if cls_4_num < cls_num_ext:
                    labels.append(int(values[1])) # save the polarity
                    with open(user_file, 'r') as inp: # save the
                        texts.append(inp.read())
                cls_4_num += 1


print('Found %s texts.' % len(texts))

model = gensim.models.Word2Vec.load("test_all.model")
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
# print(model.wv['i'])
# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = model.wv[model.wv.index2word[i]]
    # print (word, embedding_vector)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# embedding_matrix = np.zeros((num_words, 200))
# for i in range(len(model.wv.vocab)):
#     embedding_vector = model.wv[model.wv.index2word[i]]
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
print(np.shape(embedded_sequences))

x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
print (np.shape(x))
preds = Dense(5, activation='softmax')(x)
# preds = softmax(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
#rmsprop
model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
validation_data=(x_val, y_val))
