import operator
import csv
import nltk


def give_freq_counts(filename, separator, column):
    input_text = []
    frequency = {}

    with open(filename, 'r') as file_obj:
        for line in csv.reader(file_obj, delimiter=separator, skipinitialspace=True, quoting=csv.QUOTE_NONE):
            if line:
                input_word = line[column]
                input_text.append(input_word)

    input_vocab = set(input_text)
    freq_dist = nltk.FreqDist(input_text)
    for word in input_vocab:
        frequency[word] = freq_dist[word]

    sorted_freq = dict(sorted(frequency.items(), key=operator.itemgetter(1), reverse=True))
    return sorted_freq


def give_vocabulary(filename, separator='\t', column=1):
    input_text = []
    with open(filename, 'r') as file_obj:
        for line in csv.reader(file_obj, delimiter=separator, skipinitialspace=True):
            if line:
                input_word = line[column]
                input_text.append(input_word)

    input_vocabulary = set(input_text)
    return input_vocabulary


