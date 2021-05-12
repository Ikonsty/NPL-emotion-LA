import pandas as pd

from sklearn import neighbors

from nltk import word_tokenize
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_and_tokenize(data):
    # remove html markup
    data = re.sub("(<.*?>)", "", data)
    # remove urls
    data = re.sub(r'http\S+', '', data)
    # remove hashtags and @names
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    # remove punctuation and non-ascii digits
    data = re.sub("(\\W|\\d)", " ", data)
    # remove whitespace
    data = data.strip()
    # tokenization with nltk
    data = word_tokenize(data)
    # stemming with nltk
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]
    return stem_data


df_train = pd.read_csv('data/data_train.csv')
df_test = pd.read_csv('data/data_test.csv')

x_train = df_train['Text']
x_test = df_test['Text']

y_train = df_train['Emotion']
y_test = df_test['Emotion']

class_names = ['joy', 'sadness', 'anger', 'neutral', 'fear']
data = pd.concat([df_train, df_test])

# print('size of training set: %s' % (len(df_train['Text'])))
# print('size of validation set: %s' % (len(df_test['Text'])))
# print(data.Emotion.value_counts())


vector = TfidfVectorizer(tokenizer=preprocess_and_tokenize, lowercase=True,
                         sublinear_tf=True, ngram_range=(1, 2))

vector.fit_transform(data.Text)

x_train_vector = vector.transform(x_train)
x_test_vector = vector.transform(x_test)

K = 153
knn = neighbors.KNeighborsClassifier(n_neighbors=K,
                                     weights='distance',
                                     algorithm='brute',
                                     metric='cosine')

knn.fit(x_train_vector, y_train)
predicted = knn.predict(x_test_vector)

res = 0
for i in range(len(y_test)):
    if y_test[i] == predicted[i]:
        res += 1
accuracy = res / len(y_test)
print('Accuracy of knn with K =', K, ': ', accuracy)
