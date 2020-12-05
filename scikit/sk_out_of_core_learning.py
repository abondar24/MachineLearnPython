import pyprind
import pandas as pd
import os
import numpy as np
import sys
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

nltk.download('stopwords')
stop = stopwords.words('english')


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)  # remove html
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emotions).replace('-', '')  # remove non-world chars
    #  and convert to lower-case
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='movie_data.csv')

pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    x_train, y_train = get_minibatch(doc_stream, size=1000)
    if not x_train:
        break
    x_train = vect.transform(x_train)
    clf.partial_fit(x_train, y_train, classes=classes)
    pbar.update()

x_test, y_test = get_minibatch(doc_stream, size=5000)
x_test = vect.transform(x_test)

print('Accuracy: %.3f' % clf.score(x_test,  y_test))