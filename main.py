import glob
import math
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

        for word in words:
            if word not in stop_words:
                total_words_count = total_words_count + 1
                vocabulary[word] = 0
                words_frequency[word] = words_frequency.get(word,0) + 1

    return words_frequency, total_words_count


with open('data/English-Stop-Words.txt', 'r') as file:
    stop_words = file.read()

training_spam_documents = glob.glob("data/train/train-spam-*.txt")
training_ham_documents = glob.glob("data/train/train-ham-*.txt")
ham_prior = len(training_ham_documents)/(len(training_ham_documents) + len(training_spam_documents))
spam_prior = len(training_spam_documents)/(len(training_ham_documents) + len(training_spam_documents))


vocabulary = {}
delta = 0.5

spam_words_frequency, spam_words_count = load_words(training_spam_documents, stop_words, vocabulary)
ham_words_frequency, ham_words_count = load_words(training_ham_documents, stop_words, vocabulary)

vocabulary_size = len(vocabulary)
conditional_prob = np.zeros((2, vocabulary_size))
model = []

#sorting the vocabulary
vocabulary_sorted = sorted(vocabulary.keys())

#smoothing
spam_words_count = spam_words_count + delta * vocabulary_size
ham_words_count = ham_words_count + delta * vocabulary_size

#Conditional Probability
word_index = 0
for word in vocabulary_sorted:
    vocabulary[word] = word_index
    conditional_prob[0, word_index] = ham_words_frequency.get(word, 0) + delta / ham_words_count
    conditional_prob[1, word_index] = spam_words_frequency.get(word, 0) + delta / spam_words_count

    model_line = str(word_index+1)+"  "+word+"  "+ str(ham_words_frequency.get(word, 0))+"  " +\
                    str(conditional_prob[0, word_index])+"  " + str(spam_words_frequency.get(word, 0)) + "  " + \
                    str(conditional_prob[1, word_index])

    model.append(model_line)
    word_index = word_index + 1

# writing model.txt for Task 1
with open('data/model.txt', 'w') as file:
    for line in model:
        file.write(line + '\n')

#Testing
testing_documents = glob.glob("data/test/*.txt")
testing_result = []
document_count = 1


for document_path in testing_documents:
    print(document_path)
    with open(document_path,  'r') as file:
        data = file.read()
    document = document_path[10:]
    data = data.lower()
    words = re.split('[^a-zA-Z]', data)

    score_ham = math.log10(ham_prior)
    score_spam = math.log10(spam_prior)
    for word in words:
        if word in vocabulary:
            word_index = vocabulary[word]
            score_ham = score_ham + math.log10(conditional_prob[0, word_index])
            score_spam = score_spam + math.log10(conditional_prob[1, word_index])

    if score_ham > score_spam:
        predicted_class = "ham"
    else:
        predicted_class = "spam"

    if (predicted_class == document[5:8] or
            predicted_class == document[5:9]):
        classification = "right"
    else:
        classification = "wrong"

    result_line = str(document_count) + "  " + document + "  " + predicted_class + "   " + str(score_ham)\
                  +"  "+ str(score_spam) +"  "+ classification
    print(result_line)
    testing_result.append(result_line)

    document_count = document_count + 1

with open('data/baseline-result.txt ', 'w') as file:
    file.write(result_line + '\n')