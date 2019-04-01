import matplotlib.pyplot as plt
import datetime
from classifier import Classifier


def plot_graph(x, y, title, x_label, y_label):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y, color='blue')
    plt.show()


def word_filtering(filter_type, start, end, step_size):
    x = []
    y = []
    time = []
    for max_frequency in range(start, end, step_size):
        start_time = datetime.datetime.now()
        classifier = Classifier()
        classifier.load_vocabulary("data/train/train-ham-*.txt", "data/train/train-spam-*.txt")
        if filter_type == 'words':
            classifier.infrequent_word_filtering(max_frequency)
        else:
            classifier.infrequent_word_filtering_percentage(max_frequency)

        classifier.build_model(None)
        accuracy = classifier.test_model("data/test/*.txt", None)
        end_time = datetime.datetime.now()
        print('For word filtering frequency = ' + str(max_frequency) + ',   Accuracy =' + str(
            accuracy) + '   Time taken =' + str((end_time - start_time).total_seconds()))
        x.append(max_frequency)
        y.append(accuracy)
        time.append((end_time - start_time).total_seconds())

    plt.title('Accuracy vs Word Filter Frequency')
    plt.xlabel('Word Filter Frequency')
    plt.ylabel('Accuracy')
    plt.plot(x, y, color='blue')
    plt.show()

    plt.title('Time taken vs Word Filter Frequency')
    plt.xlabel('Word Filter Frequency')
    plt.ylabel('Time taken')
    plt.plot(x, time, color='blue')
    plt.show()


def smoothing():
    x = []
    y = []
    time = []
    i = 0
    delta = 0

    while delta <= 1.0:
        start_time = datetime.datetime.now()
        classifier = Classifier()
        classifier.load_vocabulary("data/train/train-ham-*.txt", "data/train/train-spam-*.txt")
        classifier.perform_smoothing(delta)
        classifier.build_model(None)
        accuracy = classifier.test_model("data/test/*.txt", None)
        end_time = datetime.datetime.now()
        print('For smoothing = ' + str(delta) + ',   Accuracy =' + str(accuracy) + '   Time taken =' + str(
            (end_time - start_time).total_seconds()))
        x.append(delta)
        y.append(accuracy)
        i += 1
        delta = round(delta + 0.1, 1)
        time.append((end_time - start_time).total_seconds())

    title = 'Accuracy vs Smoothing(delta)'
    x_label = 'Smoothing(delta)'
    y_label = 'Accuracy'
    plot_graph(x, y, title, x_label, y_label)

    title = 'Time taken vs Smoothing(delta)'
    y_label = 'Time taken'
    plot_graph(x, time, title, x_label, y_label)


def experiments_one_to_Three(experiment_no):
    start_time = datetime.datetime.now()
    classifier = Classifier()
    if experiment_no == 2:
        classifier.load_stopwords('data/English-Stop-Words.txt')
    elif experiment_no == 3:
        classifier.word_length_filtering = True
    classifier.load_vocabulary("data/train/train-ham-*.txt", "data/train/train-spam-*.txt")
    classifier.build_model('data/stopword-model.txt')
    accuracy = classifier.test_model("data/test/*.txt", 'data/stopword-result.txt')
    end_time = datetime.datetime.now()
    print('Accuracy is:' + str(accuracy))
    print('Time taken by the Classifier (seconds):' + str((end_time - start_time).total_seconds()))
