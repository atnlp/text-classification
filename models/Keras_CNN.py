# encoding: utf-8

from keras.layers import Input, Dense, Conv1D, Dropout, MaxPooling1D, Flatten, Embedding, concatenate
from keras.models import Model


class Keras_CNN():

    maxlen = 200
    max_features = 50000
    embed_size = 300
    label_seq = Input(shape=[maxlen], name='x_seq')
    emb_label = Embedding(max_features, embed_size)(label_seq)

    convs = []
    filter_sizes = [3, 4, 5]
    for fsz in filter_sizes:
        conv = Conv1D(filters=200, kernel_size=fsz, activation='tanh')(emb_label)
        pool = MaxPooling1D(maxlen - fsz + 1)(conv)
        pool = Flatten()(pool)
        convs.append(pool)

    merge = concatenate(convs, axis=1)

    out = Dropout(0.2)(merge)
    output = Dense(128, activation='relu')(out)

    output = Dense(1, activation='sigmoid')(output)

    model = Model([label_seq], output)