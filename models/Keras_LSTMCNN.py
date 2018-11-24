# encoding: utf-8
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D, Dropout


class Keras_LSTMCNN():
    # Convolution  卷积
    filter_length = 5  # 滤波器长度
    nb_filter = 64  # 滤波器个数
    pool_length = 4  # 池化长度

    # LSTM
    lstm_output_size = 70  # LSTM 层输出尺寸

    # 设定参数
    max_features = 50000   # 词汇表大小
    # cut texts after this number of words (among top max_features most common words)
    # 裁剪文本为 maxlen 大小的长度（取最后部分，基于前 max_features 个常用词）
    maxlen = 200
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))  # 词嵌入层
    model.add(Dropout(0.25))  # Dropout层

    # 1D 卷积层，对词嵌入层输出做卷积操作
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    # 池化层
    model.add(MaxPooling1D(pool_length=pool_length))
    # LSTM 循环层
    model.add(LSTM(lstm_output_size))
    # 全连接层，只有一个神经元，输入是否为正面情感值
    model.add(Dense(1))
    model.add(Activation('sigmoid'))  # sigmoid判断情感