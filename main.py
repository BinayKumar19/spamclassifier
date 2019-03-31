import glob
import re


def load_words(documents, stop_words):
    vocabulary = {}
    for document in documents:
        with open(document, 'r') as file:
            data = file.read()

        words = re.split('[^a-zA-Z]', data)
        for word in words:
            if word not in stop_words:
                if word not in vocabulary:
                    vocabulary[word] = 1
                else:
                    vocabulary[word] = vocabulary[word] + 1

    return vocabulary


with open('data/English-Stop-Words.txt', 'r') as file:
    stop_words = file.read()

training_spam_documents = glob.glob("data/train/train-spam-*.txt")
training_ham_documents = glob.glob("data/train/train-ham-*.txt")

spam_dictionary = load_words(training_spam_documents, stop_words)
ham_dictionary = load_words(training_ham_documents, stop_words)




