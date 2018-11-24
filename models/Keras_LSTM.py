# encoding: utf-8
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb


class Keras_LSTM():

    # 设定参数
    max_features = 50000   # 词汇表大小
    # cut texts after this number of words (among top max_features most common words)
    # 裁剪文本为 maxlen 大小的长度（取最后部分，基于前 max_features 个常用词）
    maxlen = 200

    model = Sequential()
    # 嵌入层，每个词维度为128
    model.add(Embedding(max_features, 128, dropout=0.2))
    # LSTM层，输出维度128，可以尝试着换成 GRU 试试
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
    model.add(Dense(1))   # 单神经元全连接层
    model.add(Activation('sigmoid'))   # sigmoid 激活函数层

    # model.summary()   # 模型概述
