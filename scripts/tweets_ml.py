import pandas as pd
import numpy as np
from nltk import word_tokenize, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import _pickle as cPickle
import os


def open_csv_file(file_name, seperator, nameslist):
    '''df = pd.read_csv('data.csv',
                    sep=',',
                    header=None,
                    names=['label', 'id','datetime','source','username','tweet'])

    return : DataFrame'''

    df = pd.read_csv(file_name,
                     sep=seperator,
                     header=None,
                     names=nameslist)
    return df


def stem_data(df, value_label):
    stemmer = PorterStemmer()
    df[value_label] = df.tweet.map(lambda x: x.lower())
    df[value_label] = df.tweet.str.replace(r'[^\w\s]', '')
    df[value_label] = df[value_label].apply(word_tokenize)
    df[value_label] = df[value_label].apply(
        lambda x: [stemmer.stem(y) for y in x])

    # This converts the list of words into space-separated strings
    df[value_label] = df[value_label].apply(lambda x: ' '.join(x))
    return df


def convert_to_tfidf(df, value_label, btransformer):
    if btransformer == None:
        bow_transformer = CountVectorizer().fit(df[value_label])
    else:
        bow_transformer = btransformer

    x = bow_transformer.transform(df[value_label])
    '''count_vect = CountVectorizer()
        counts = count_vect.fit_transform(df[value_label])
        transformer = TfidfTransformer().fit(counts)
        counts = transformer.transform(counts)'''
    return bow_transformer, x


def create_model(counts, df, class_label):
    X_train, _, y_train, _ = train_test_split(
        counts, df[class_label], test_size=0.1, random_state=69)
    model = MultinomialNB().fit(X_train, y_train)
    return model


def predect_from_df(csv_name, value_label):

    df = open_csv_file(csv_name, ",", [value_label])

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"), "rb") as f:
        model = cPickle.load(f, encoding='latin1')
        bow_transformer = cPickle.load(f, encoding='latin1')

    stemed_df = stem_data(df, value_label)
    _, counts_vector = convert_to_tfidf(
        stemed_df, value_label, bow_transformer)
    predicted = model.predict(counts_vector)
    return predicted


def new_model():
    df = open_csv_file('data.csv', ',', [
                       'label', 'id', 'datetime', 'source', 'username', 'tweet'])
    stemed_df = stem_data(df, 'tweet')
    bow_transformer, counts_vectors = convert_to_tfidf(df, 'tweet', None)
    model = create_model(counts_vectors, stemed_df, 'label')

    try:
        with open("model", mode="wb") as f:
            cPickle.dump(model, f)
            cPickle.dump(bow_transformer, f)
    except Exception as e:
        pass
