import csv
import numpy as np
import scipy as sp
from sklearn.feature_extraction import text
import nltk
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression


def give_training_data(filename, separator):
    features_train = []
    labels_train = []
    post_train = []
    post = []
    cap_tr = []
    title_tr = []
    lo_tr = []
    dig_tr = []
    with open(filename, 'r') as file_obj:
        for line in csv.reader(file_obj, delimiter=separator, skipinitialspace=True, quoting=csv.QUOTE_NONE):
            if line:
                features_train.append(line[1])
                labels_train.append(line[-1])
                cap_tr.append(line[1].isupper())
                title_tr.append(line[1].istitle())
                lo_tr.append(line[1].islower())
                dig_tr.append(line[1].isdigit())
        post.append(nltk.pos_tag(features_train))
        for elem in post:
            for i in elem:
                post_train.append(i[-1])
    return features_train, labels_train, post_train, cap_tr, title_tr, lo_tr, dig_tr


def give_test_data(filename, separator):
    input_text = []
    post_test = []
    post = []
    cap_test = []
    title_test = []
    lo_test = []
    dig_test = []
    with open(filename, 'r') as file_obj:
        for line in csv.reader(file_obj, delimiter=separator, skipinitialspace=True, quoting=csv.QUOTE_NONE):
            if line:
                input_text.append(line[1])
                cap_test.append(line[1].isupper())
                title_test.append(line[1].istitle())
                lo_test.append(line[1].islower())
                dig_test.append(line[1].isdigit())
        post.append(nltk.pos_tag(input_text))
        for elem in post:
            for i in elem:
                post_test.append(i[-1])
    return input_text, post_test, cap_test, title_test, lo_test, dig_test


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


def extract_features(features_train, features_test, post_train, post_test, cap_train, cap_test,
                     title_train, title_test, low_train, low_test, digit_tr, digit_test):
    vectorizer_1 = text.CountVectorizer(ngram_range=(1, 1))
    vectorizer_2 = text.CountVectorizer(ngram_range=(1, 1))
    training_vector = vectorizer_1.fit_transform(features_train)
    test_vector = vectorizer_1.transform(features_test)
    ptraining_vector = vectorizer_2.fit_transform(post_train)
    ptest_vector = vectorizer_2.transform(post_test)
    training_vec = sp.sparse.hstack((training_vector, ptraining_vector, csr_matrix(cap_train).T,
                                     csr_matrix(title_train).T, csr_matrix(low_train).T, csr_matrix(digit_tr).T))
    test_vect = sp.sparse.hstack((test_vector, ptest_vector, csr_matrix(cap_test).T,
                                  csr_matrix(title_test).T, csr_matrix(low_test).T, csr_matrix(digit_test).T))
    return training_vec, test_vect


def classify(training_vector, labels_train, test_vector):
    print('classifying')
    clf = LogisticRegression()
    clf.fit(training_vector, labels_train)
    print('printing')
    pred = clf.predict(test_vector)
    return pred


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data,indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


if __name__ == "__main__":

    training = 'Training.txt'
    devset = 'test.txt'
    out_file = 'output.txt'
    delim = '\t'

    f_t, l_t, p_t, c_t, t_t, lo_t, d_tr = give_training_data(training, delim)
    f_test, p_test, c_test, t_Test, l_test, d_test = give_test_data(devset, delim)
    tr_vec, test_vec = extract_features(f_t, f_test, p_t, p_test, c_t, c_test, t_t, t_Test, lo_t,
                                        l_test, d_tr, d_test)
    answer_list = classify(tr_vec, l_t, test_vec)
    a_list = answer_list.tolist()
    print_output(devset, a_list, out_file, delim)
