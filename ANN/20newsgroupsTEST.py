import tensorflow as tf
import numpy as np
import mnist
import NNutils

import scipy.sparse
from sklearn.datasets import fetch_20newsgroups
# from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_extraction.text import TfidfVectorizer
# from datetime import datetime


#20newsgroups 셋을 불러옴
dataset = fetch_20newsgroups(data_home="C:/YWK/Projects/Python/Data/20newsgroups", subset='all')    #이걸로 하지말고 train에 맞춰서 해보자.
trainset = fetch_20newsgroups(data_home="C:/YWK/Projects/Python/Data/20newsgroups", subset='train')
testset = fetch_20newsgroups(data_home="C:/YWK/Projects/Python/Data/20newsgroups", subset='test')

#벡터화
vectorizer = TfidfVectorizer(analyzer='word', max_features=1000)


vectorizer.fit(dataset.data)
train_data = vectorizer.transform(trainset.data)
train_data = scipy.sparse.csr_matrix.todense(train_data)
train_label = trainset.target
train_label =  NNutils.onehot(20, train_label)
print(train_data.shape)
print(train_data[0])

# test_data = vectorizer.fit_transform(testset.data)
# test_data = scipy.sparse.csr_matrix.todense(test_data)
# test_label = testset.target
# print(test_data[0])


