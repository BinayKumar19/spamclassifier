from experiments import experiments_one_to_Three, smoothing, word_filtering

experiment = int(input("Choose the experiment:\n1. baseline experiment: \n2. Stop-word Filtering"
                       "\n3. Word Length Filtering \n4. Infrequent Word Filtering\n5. Smoothing"))

if experiment == 1:
    print("baseline experiment")
    experiments_one_to_Three(experiment)

elif experiment == 2:
    print("baseline experiment + stop words")
    experiments_one_to_Three(experiment)

elif experiment == 3:
    print("baseline experiment + Word Length Filtering")
    experiments_one_to_Three(experiment)

elif experiment == 4:
    print("baseline experiment + Infrequent Word Filtering")
    word_filtering('words', 0, 21, 5)
    word_filtering('percentage', 5, 26, 5)

elif experiment == 5:
    print("baseline experiment + Smoothing")
    smoothing()

else:
    print('Input should be between 1-5')
