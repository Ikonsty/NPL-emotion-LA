import pandas as pd
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def bayes_classifier():
	split_data = split_vectorize_data()[0]
	x_train_vectors, x_test_vectors = split_data[0], split_data[1]
	y_train, y_test = split_data[2], split_data[3]

	nb = MultinomialNB()

	nb.fit(x_train_vectors, y_train)

	predicted = nb.predict(x_test_vectors)
	
	res = 0
	for i in range(len(y_test)):
		if y_test[i] == predicted[i]:
			res += 1
	accuracy = res / len(y_test)
	print('Accuracy of knn with K =', K, ': ', accuracy)



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

def split_vectorize_data():
	df_train = pd.read_csv('data/data_train.csv')
	df_test = pd.read_csv('data/data_test.csv')

	x_train = df_train['Text']
	x_test = df_test['Text']

	y_train = df_train['Emotion']
	y_test = df_test['Emotion']

	class_names = ['joy', 'sadness', 'anger', 'neutral', 'fear']
	data = pd.concat([df_train, df_test])

	vector = TfidfVectorizer(tokenizer=preprocess_and_tokenize,
							 lowercase=True,
							 sublinear_tf=True, ngram_range=(1, 2))

	vector.fit_transform(data.Text)

	x_train_vector = vector.transform(x_train)
	x_test_vector = vector.transform(x_test)

	return (x_train_vector, x_test_vector, y_train, y_test)
