# encoding: utf-8

import os
import re
import numpy as np
import pandas as pd


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_files(filetype):
    path = "datasets/aclImdb/"
    file_list = []
    positive_path = path + filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path+f]
    negative_path = path + filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path+f]
    print('read', filetype, 'files:', len(file_list))
    all_labels = np.array(([1] * 12500 + [0] * 12500))
    all_texts = []

    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            filelines = file_input.readlines()
            all_texts += [rm_tags(filelines[0])]

    all_texts = [z.lower().replace('\n', '') for z in all_texts]
    all_texts = [z.replace('<br />', ' ') for z in all_texts]

    all_texts = np.array(all_texts)
    # return all_labels, all_texts
    if filetype == 'train':
        return pd.DataFrame({'text': pd.Series(all_texts), 'label': pd.Series(all_labels)}).sample(frac=1)
    else:
        return pd.DataFrame({'text': pd.Series(all_texts), 'label': pd.Series(all_labels)})


