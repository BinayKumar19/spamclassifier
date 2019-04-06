from experiments import experiments_one_to_five, smoothing, word_filtering



experiment = int(input("Choose the experiment:\n1. baseline experiment: \n2. Stop-word Filtering"
                       "\n3. Word Length Filtering \n4. Infrequent Word Filtering\n5. Smoothing"
                       "\n6. baseline experiment + Infrequent Word Filtering Different Values "
                       "\n7. baseline experiment + Smoothing Different Values"))

choice = input('Do you want to use default training and test path?\n1. Yes\n2. No\n')
if choice == '2':
    train_documents_path = input('Enter path for Training Files')
    test_documents_path = input('Enter path for Testing Files')
else:
    train_documents_path = "data/train/"
    test_documents_path = "data/test/"

if experiment in range(1,4):

    experiments_one_to_five(experiment,train_documents_path,test_documents_path)

elif experiment == 4:
    print("baseline experiment + Infrequent Word Filtering")
    frequent_word_count = int(input("Enter upper limit for the word frequency to be removed from the dictionary, "
                                    "0 for No constraint: "))
    frequent_word_percentage = int(input("Enter upper limit for the word frequency percentage to be removed from the "
                                         "dictionary, 0 for No constraint: "))
    experiments_one_to_five(experiment, train_documents_path, test_documents_path, frequent_word_count, frequent_word_percentage)

elif experiment == 5:
    print("baseline experiment + Smoothing")
    smoothing_value = float(input("Please enter the smoothing value: "))
    experiments_one_to_five(experiment, train_documents_path, test_documents_path, smoothing_value)

elif experiment == 6:
    print("baseline experiment + Infrequent Word Filtering Different Values")
    word_filtering(train_documents_path, test_documents_path,'words', 0, 21, 5)
    word_filtering(train_documents_path, test_documents_path, 'percentage', 5, 26, 5)
elif experiment == 7:
    print("baseline experiment + Smoothing Different Values")
    smoothing(train_documents_path, test_documents_path,)
else:
    print('Input should be between 1-7')
