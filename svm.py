from sklearn import svm
import csv
import numpy as np
from sklearn.feature_extraction import text


def give_training_data(filename, separator):
    features_train = []
    labels_train = []

    with open(filename, 'r') as file_obj:
        for line in csv.reader(file_obj, delimiter=separator, skipinitialspace=True, quoting=csv.QUOTE_NONE):
            if line:
                features_train.append(line[1])
                labels_train.append(line[-1])

    return features_train, labels_train


def give_test_data(filename, separator):
    input_text = []

    with open(filename, 'r') as file_obj:
        for line in csv.reader(file_obj, delimiter=separator, skipinitialspace=True, quoting=csv.QUOTE_NONE):
            if line:
                input_word = line[1]
                input_text.append(input_word)
    return input_text


def print_output(dev_set, answers, outfile, separator):
    with open(outfile, "w", newline='') as output:
        with open(dev_set, "r") as get_input:
            index = 0
            writer = csv.writer(output, delimiter=separator, skipinitialspace=True)
            for row in csv.reader(get_input, delimiter=separator, skipinitialspace=True, quoting=csv.QUOTE_NONE):
                if row:
                    row.append(answers[index])
                    writer.writerow(row)
                    index = index + 1
                else:
                    writer.writerow('\n'.strip())
    return output


def classify(features_train, labels_train, features_test):
    vectorizer = text.TfidfVectorizer(ngram_range=(1, 2))
    training_vector = vectorizer.fit_transform(features_train)
    test_vector = vectorizer.transform(features_test)

    clf = svm.SVC(10000, kernel='linear')
    clf.fit(training_vector, labels_train)
    pred = clf.predict(test_vector)
    return pred

if __name__ == "__main__":

    training = 'Training.txt'
    devset = 'Test.txt'
    out_file = 'output.txt'
    delim = '\t'

    f_t, l_t = give_training_data(training, delim)
    f_test = give_test_data(devset, delim)
    a_1 = np.array(f_t)
    a_2 = np.array(l_t)
    a_3 = np.array(f_test)
    answer_list = classify(a_1, a_2, a_3)
    a_list = answer_list.tolist()
    print_output(devset, a_list, out_file, delim)
