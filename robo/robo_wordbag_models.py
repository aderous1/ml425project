
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers

manual = pd.read_csv('data/all_filtered_randosort_cropped_manual_v1.csv')

print(f"Manual labeled data: {manual.columns} \n{manual}")

vectorizer = CountVectorizer(
    min_df=0, lowercase=True,
    strip_accents='unicode',
    ngram_range=(1, 3)
)

related = manual[manual['covid related?'] > 0]

print(f"Only related: \n{related}")

no_urls = re.compile(r'https?:\S+')

sentences = related['Tweet Text'].str.replace(no_urls, '')
reliability = related['Reliability Score']

vectorizer.fit(sentences)

print(f"vectorizer size: {len(vectorizer.vocabulary_)}")

sents_train, sents_test, cat_train, cat_test = train_test_split(
    sentences, reliability, test_size=0.1
)

x_train = vectorizer.transform(sents_train)
x_test = vectorizer.transform(sents_test)

print(f"Training count: {x_train.shape}")

# NOTE actual ML approaches now

## let's use LogisticRegression as a baseline to compare other approaches to

lr_classifier = LogisticRegression()

lr_classifier.fit(x_train, cat_train)
score = lr_classifier.score(x_test, cat_test)
print(f"LogisticRegression score: {score}")

## try Keras sequential models:

input_dim = x_train.shape[1]

k_seq_model = Sequential()
k_seq_model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
k_seq_model.add(layers.Dense(1, activation='sigmoid'))

k_seq_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(f"Keras sequential model: \n{k_seq_model}")
