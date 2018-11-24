# encoding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from data import read_files


def vectorization(vector_type):

    train = read_files('train')
    test = read_files('test')

    if vector_type == 'CountVectorizer':
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=2000)
    elif vector_type == 'TfidfVectorizer':
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = HashingVectorizer(n_features=200)

    vectorizer.fit(train['text'])

    x_train = vectorizer.transform(train['text'])
    y_train = train['label'].values

    x_test = vectorizer.transform(test['text'])
    y_test = test['label'].values

    return x_train, y_train, x_test, y_test


# 使用keras的文本预处理函数进行Embedding之前的预处理
def word_vectors():

    train = read_files('train')
    test = read_files('test')

    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(train['text'])

    train_seq = tokenizer.texts_to_sequences(train['text'])
    test_seq = tokenizer.texts_to_sequences(test['text'])

    x_train = sequence.pad_sequences(train_seq, maxlen=200)  # shape  (25000, 200)
    y_train = train['label']
    x_test = sequence.pad_sequences(test_seq, maxlen=200)  # shape (25000, 200)
    y_test = test['label']
    return x_train, y_train.values, x_test, y_test.values


