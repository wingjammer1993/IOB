import numpy


def construct_local_transition(state_sequence, master_a):
    a = numpy.zeros((len(state_sequence), len(state_sequence)))
    for row, tag_row in enumerate(state_sequence):
        for column, tag_column in enumerate(state_sequence):
            required_row = master_a[tag_row]
            tag_tag_probability = required_row[tag_column]
            a[row][column] = tag_tag_probability
    return a


def construct_local_emission(state_sequence, observation_sequence, vocabulary, master_b):
    b = numpy.zeros((len(state_sequence), len(observation_sequence)))
    for row, tag_row in enumerate(state_sequence):
        for column, tag_column in enumerate(observation_sequence):
            required_row = master_b[tag_row]
            if tag_column in vocabulary:
                word_tag_probability = required_row[tag_column]
                b[row][column] = word_tag_probability
            else:
                b[row][column] = 0
    return b


def construct_local_pie_start(state_sequence, master_pie_start):
    pie_1 = numpy.zeros(len(state_sequence))
    for row, tag_row in enumerate(state_sequence):
        end_tag_probability = master_pie_start[tag_row]
        pie_1[row] = end_tag_probability
    return pie_1


def construct_local_pie_end(state_sequence, master_pie_end):
    pie_2 = numpy.zeros(len(state_sequence))
    for row, tag_row in enumerate(state_sequence):
            required_row = master_pie_end[tag_row]
            end_tag_probability = required_row['<e>']
            pie_2[row] = end_tag_probability
    return pie_2

