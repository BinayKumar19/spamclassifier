import glob
import math
import re
import numpy as np


class Classifier:

    def __init__(self):
        self.stop_words = []
        self.vocabulary = {}
        self.conditional_prob = None

        self.spam_words_frequency = {}
        self.spam_words_count = 0
        self.spam_words_count_initial = 0
        self.spam_prior = 0

        self.ham_words_frequency = {}
        self.ham_words_count = 0
        self.ham_words_count_initial = 0
        self.ham_prior = 0

        self.delta = 0.0
        self.word_length_filtering = False

    def load_words(self, documents):
        words_frequency = {}
        total_words_count = 0
        for document in documents:
            with open(document, 'r') as file:
                data = file.read()

            data = data.lower()
            words = re.split('[^a-zA-Z]', data)

            for word in words:
                if word not in self.stop_words:
                    if self.word_length_filtering and len(word) not in range(3, 9):
                        continue
                    total_words_count = total_words_count + 1
                    self.vocabulary[word] = 0
                    words_frequency[word] = words_frequency.get(word, 0) + 1

        return words_frequency, total_words_count

    def file_write(self, file_path, data):
        with open(file_path, 'w') as file:
            for line in data:
                file.write(line + '\n')

    def load_stopwords(self, file_path):
        with open(file_path, 'r') as file:
            self.stop_words = file.read()

    def load_vocabulary(self, ham_documents_path, spam_documents_path):
        training_ham_documents = glob.glob(ham_documents_path)
        training_spam_documents = glob.glob(spam_documents_path)
        self.ham_prior = len(training_ham_documents) / (len(training_ham_documents) + len(training_spam_documents))
        self.spam_prior = len(training_spam_documents) / (len(training_ham_documents) + len(training_spam_documents))
        self.spam_words_frequency, self.spam_words_count_initial = self.load_words(training_spam_documents)
        self.ham_words_frequency, self.ham_words_count_initial = self.load_words(training_ham_documents)
        self.spam_words_count = self.spam_words_count_initial
        self.ham_words_count = self.ham_words_count_initial

    def perform_smoothing(self, delta):
        self.delta = delta
        self.spam_words_count = self.spam_words_count_initial + self.delta * len(self.vocabulary)
        self.ham_words_count = self.ham_words_count_initial + self.delta * len(self.vocabulary)

    def build_model(self, output_file_path):
        vocabulary_size = len(self.vocabulary)
        model = []
        self.conditional_prob = np.zeros((2, vocabulary_size))

        # sorting the vocabulary
        vocabulary_sorted = sorted(self.vocabulary.keys())

        # Conditional Probability
        word_index = 0
        for word in vocabulary_sorted:
            self.vocabulary[word] = word_index
            self.conditional_prob[0, word_index] = self.ham_words_frequency.get(word,
                                                                                0) + self.delta / self.ham_words_count
            self.conditional_prob[1, word_index] = self.spam_words_frequency.get(word,
                                                                                 0) + self.delta / self.spam_words_count

            model_line = str(word_index + 1) + "  " + word + "  " + str(self.ham_words_frequency.get(word, 0)) + "  " + \
                         str(self.conditional_prob[0, word_index]) + "  " + str(
                self.spam_words_frequency.get(word, 0)) + "  " + \
                         str(self.conditional_prob[1, word_index])

            model.append(model_line)
            word_index = word_index + 1

        if output_file_path != None:
            self.file_write(output_file_path, model)

    def test_model(self, test_documents_path, output_file_path):
        testing_documents = glob.glob(test_documents_path)
        testing_result = []
        document_count = 0
        accouracy_count = 0
        non_accouracy_count = 0

        for document_path in testing_documents:
            # print(document_path)
            document_count = document_count + 1
            with open(document_path, 'r', encoding='utf-8', errors="surrogateescape") as file:
                data = file.read()
            document = document_path[10:]
            data = data.lower()
            words = re.split('[^a-zA-Z]', data)

            score_ham = math.log10(self.ham_prior)
            score_spam = math.log10(self.spam_prior)
            for word in words:
                if word in self.vocabulary:
                    word_index = self.vocabulary[word]
                    conditional_prob_ham = self.conditional_prob[0, word_index]
                    conditional_prob_spam = self.conditional_prob[1, word_index]
                    score_ham = score_ham + (math.log10(conditional_prob_ham) if conditional_prob_ham > 0 else 0)
                    score_spam = score_spam + (math.log10(conditional_prob_spam) if conditional_prob_spam > 0 else 0)

            if score_ham > score_spam:
                predicted_class = "ham"
            else:
                predicted_class = "spam"

            if (predicted_class == document[5:8] or
                    predicted_class == document[5:9]):
                classification = "right"
                accouracy_count = accouracy_count + 1
            else:
                classification = "wrong"
                non_accouracy_count = non_accouracy_count + 1

            result_line = str(document_count) + "  " + document + "  " + predicted_class + "   " + str(score_ham) \
                          + "  " + str(score_spam) + "  " + classification
            # print(result_line)
            testing_result.append(result_line)

        # print(accouracy_count)
        # print(non_accouracy_count)
        # print(document_count)
        accuracy = (accouracy_count / document_count) * 100

        if output_file_path != None:
            self.file_write(output_file_path, testing_result)

        return round(accuracy, 2)

    def infrequent_word_filtering(self, max_frequency):

        for word in list(self.vocabulary.keys()):
            spam_frequency = self.spam_words_frequency.get(word, 0)
            ham_frequency = self.ham_words_frequency.get(word, 0)
            total_frequency = ham_frequency + spam_frequency
            if total_frequency in range(1, max_frequency + 1):
                self.vocabulary.pop(word)

    def infrequent_word_filtering_percentage(self, max_frequency_percentage):
        vocabulary_temp = {}
        for word in list(self.vocabulary.keys()):
            spam_frequency = self.spam_words_frequency.get(word, 0)
            ham_frequency = self.ham_words_frequency.get(word, 0)
            total_frequency = ham_frequency + spam_frequency
            vocabulary_temp[word] = total_frequency

        vocabulary_temp = sorted(vocabulary_temp.items(), reverse=True, key=lambda kv: (kv[1], kv[0]))
        max_frequency = (max_frequency_percentage * len(self.vocabulary)) / 100

        word_count = 0
        for word, frequency in vocabulary_temp:
            word_count += 1
            if word_count > max_frequency:
                break
            else:
                self.vocabulary.pop(word)
