import pyprind
import pandas as pd
import os
import numpy as np
import sys
import re
import nltk

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)  # remove html
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emotions).replace('-', '') # remove non-world chars
    #  and convert to lower-case
    return text


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


pbar = pyprind.ProgBar(50000, stream=sys.stdout)
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()

# need to download  imdb dataset manually
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = 'aclImdb\%s\%s' % (s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding="ISO-8859-1") as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()

df.columns = ['review', 'sentiment']
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False)

df = pd.read_csv('movie_data.csv', encoding="ISO-8859-1")
print(df.head(3))

count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining and the weather is sweet'])

print('Bagging concept')
print('Docs\n', docs)
bag = count.fit_transform(docs)
print('Count vocab\n',count.vocabulary_)
print('Bag in array\n', bag.toarray())

tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

# cleaning text data
df['review'] = df['review'].apply(preprocessor)

nltk.download('stopwords')
stop = stopwords.words('english')

x_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
x_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
                'vect__stop_words': [stop, None],
               'vect_tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}
             ]
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1)
gs_lr_tfidf.fit(x_train, y_train)

print('Best parameter set: %s' % gs_lr_tfidf.best_params_)
print('CV accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test accuracy: %.3f' % clf.score(x_test, y_test))
