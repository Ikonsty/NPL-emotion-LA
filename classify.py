import pickle


def classify(message):
    filename = 'models/tfidf_svm.sav'
    model = pickle.load(open(filename, 'rb'))

    return model.predict([message])[0]

if __name__ == "__main__":
    print(classify('stupid idiot!'))
