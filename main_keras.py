# encoding: utf-8
from util import data_preprocess
import models


# Keras模型搭建代码参考https://gaussic.github.io/2017/03/03/imdb-sentiment-classification/
def main():
    batch_size = 64  # 批数据量大小
    model = getattr(models, 'Keras_LSTMCNN')().model
    x_train, y_train, x_test, y_test = data_preprocess.word_vectors()
    # print(x_train.shape)
    # print(y_train.shape)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              validation_split=0.1,
              batch_size=batch_size,
              epochs=3)


if __name__ == '__main__':
    main()
