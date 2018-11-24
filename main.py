# encoding: utf-8
from models.ml_models import *
from sklearn.metrics import classification_report
from util import data_preprocess


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('vector_type', type=str, help='a number')
    # args = parser.parse_args()

    vector_type = 'CountVectorizer'
    x_train, y_train, x_test, y_test = data_preprocess.vectorization(vector_type)

    model = clf_lg()
    model.fit(x_train, y_train)

    print(classification_report(y_train.values, model.predict(x_train)))
    print(classification_report(y_test.values, model.predict(x_test)))


if __name__ == '__main__':
    main()






