import glob
import re
import numpy as np


def load_words(documents, stop_words, vocabulary):
    words_frequency = {}
    total_words_count = 0
    for document in documents:
        with open(document, 'r') as file:
            data = file.read()

        data = data.lower()
        words = re.split('[^a-zA-Z]', data)
        if 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaazgvzywaaaaaaaaausuvdidyxoty' in words:
            print(document)
        for word in words:
            if word not in stop_words:
                total_words_count = total_words_count + 1
                vocabulary.add(word)
                words_frequency[word] = words_frequency.get(word,0) + 1

    return words_frequency, total_words_count


with open('data/English-Stop-Words.txt', 'r') as file:
    stop_words = file.read()

training_spam_documents = glob.glob("data/train/train-spam-*.txt")
training_ham_documents = glob.glob("data/train/train-ham-*.txt")
ham_prior = len(training_ham_documents)/(len(training_ham_documents) + len(training_spam_documents))
spam_prior = len(training_spam_documents)/(len(training_ham_documents) + len(training_spam_documents))


vocabulary = set()
delta = 0.5

spam_words_frequency, spam_words_count = load_words(training_spam_documents, stop_words, vocabulary)
ham_words_frequency, ham_words_count = load_words(training_ham_documents, stop_words, vocabulary)

#sorting the vocabulary
vocabulary = sorted(vocabulary)
vocabulary_size = len(vocabulary)

conditional_prob = np.zeros((2, vocabulary_size))

#smoothing
spam_words_count = spam_words_count + delta * vocabulary_size
ham_words_count = ham_words_count + delta * vocabulary_size
model = []

#Conditional Probability
for j in range(0, vocabulary_size):
    word = vocabulary[j]

    conditional_prob[0, j] = spam_words_frequency.get(word, 0) + delta / spam_words_count
    conditional_prob[1, j] = ham_words_frequency.get(word, 0) + delta / ham_words_count

    model_line = str(j+1)+"  "+word+"  "+ str(ham_words_frequency.get(word, 0))+"  " +\
                    str(conditional_prob[1, j])+"  " + str(spam_words_frequency.get(word, 0)) + "  " + \
                    str(conditional_prob[0, j])

    model.append(model_line)

with open('data/model.txt', 'w') as file:
    for line in model:
        file.write(line + '\n')

