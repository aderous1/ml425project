
import re

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from keras.models import Sequential
from tensorflow.keras.models import Sequential
# from keras import layers
from tensorflow.keras import layers

# function stolen from https://realpython.com/python-keras-text-classification/
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

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

x_train = vectorizer.transform(sents_train).todense()
x_test = vectorizer.transform(sents_test).todense()

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
k_seq_model.add(layers.Dense(2, input_dim=input_dim, activation='relu'))
k_seq_model.add(layers.Dense(1, activation='sigmoid'))

k_seq_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(f"Keras sequential model:")
k_seq_model.summary()

k_seq_hist = k_seq_model.fit(
    x_train, cat_train,
    epochs=20,
    verbose=True,
    validation_data=(x_test, cat_test),
    batch_size=5
)

loss, acc = k_seq_model.evaluate(x_train, cat_train, verbose=False)
print(f"k_seq training accuracy: {acc} with loss {loss}")
loss, acc = k_seq_model.evaluate(x_test, cat_test, verbose=False)
print(f"k_seq test set accuracy: {acc} with loss {loss}")

plot_history(k_seq_hist)
plt.show()
