import pandas as  pd
import numpy as np
from nltk import word_tokenize, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

stemmer = PorterStemmer()

df = pd.read_table('SMSSpamCollection',
        sep='\t',
        header=None,
        names=['label', 'message'])

df['label'] = df.label.map({'ham':0, 'spam' : 1})
df['message'] = df.message.map(lambda x: x.lower())
df['message'] = df.message.str.replace('[^\w\s]', '')
df['message'] = df['message'].apply(word_tokenize)
df['message'] = df['message'].apply(lambda x : [stemmer.stem(y) for y in x])

# This converts the list of words into space-separated strings
df['message'] = df['message'].apply(lambda x: ' '.join(x))

print(type(df['message']))
count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['message'])
transformer = TfidfTransformer().fit(counts)
counts = transformer.transform(counts)

X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=69)
model = MultinomialNB().fit(X_train, y_train)

predicted = model.predict(X_test)


org_test = count_vect.inverse_transform(X_test)

q = "\n".join([' '.join(y) for x, y in enumerate(org_test) if predicted[x]==1])
