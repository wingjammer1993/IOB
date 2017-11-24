import numpy


# viterbi[] = viterbi state chart
# observation_sequence = list containing sentence to be tagged
# state_sequence = list containing possible tags of sentence
# a[] = Transition Probability Matrix for the observation_sequence
# b[] = Emission Probability Matrix for the observation_sequence
# backtrace[] = stores the argmax for each state

def viterbi_decode(observation_sequence, state_sequence, a, b, pie_1, pie_2):

    rows = len(state_sequence)
    columns = len(observation_sequence)
    viterbi = numpy.zeros((rows, columns))
    backtrace = numpy.zeros((rows, columns))
    best_path = numpy.zeros(columns)

    # Initialization - we fill the first column with <s> to tag probabilities.

    for s, state in enumerate(state_sequence):
            viterbi[s][0] = pie_1[s]*b[s][0]
            backtrace[s][0] = 0

    # Recursion - we will fill the remaining columns in the state chart

    for t, observation in enumerate(observation_sequence):
        if t != 0:
            for s, state in enumerate(state_sequence):
                func = []
                func_max = []
                for s_iter, state_i in enumerate(state_sequence):
                    v1 = viterbi[s_iter][t-1]
                    v2 = a[s_iter][s]
                    v3 = b[s][t]
                    func.append(viterbi[s_iter][t-1]*a[s_iter][s]*b[s][t])
                    func_max.append(viterbi[s_iter][t - 1] * a[s_iter][s])
                viterbi[s][t] = numpy.amax(func)
                backtrace[s][t] = numpy.argmax(func_max)

    # Termination - we will fill the remaining columns in the state chart

    end_col = len(observation_sequence) - 1
    func_max = []
    for s_iter, state_i in enumerate(state_sequence):
        v1 = viterbi[s_iter][end_col]
        v2 = pie_2[s_iter]
        func_max.append(viterbi[s_iter][end_col] * pie_2[s_iter])

    best_score = numpy.amax(func_max)
    start_backtrace = numpy.argmax(func_max)

    # Backtracking

    best_path[end_col] = start_backtrace
    for col in range(end_col-1, -1, -1):
        use_index = int(best_path[col+1])
        best_path[col] = backtrace[use_index][col+1]

    answer_string = []
    for index in range(0, len(observation_sequence), 1):
        tag = best_path[int(index)]
        answer_string.append(state_sequence[int(tag)])

    return answer_string


if __name__ == "__main__":

    os = ['Janet', 'will', 'back', 'the', 'bill']
    ss = ['NNP', 'MD', 'VB', 'JJ', 'NN', 'RB', 'DT']

    a_1 = numpy.zeros((7, 7))
    b_1 = numpy.zeros((7, 5))

    pie_1 = [0.2767, 0.0006, 0.0031, 0.0453, 0.0449, 0.0510, 0.2026]
    pie_2 = [0.0001, 0.0001, 0.0001, 0.0001, 0.0101, 0.0001, 0.0001]

    a_1 = [[0.38000, 0.01100, 0.00090, 0.0084, 0.0584, 0.0090, 0.0025],
           [0.00008, 0.00020, 0.79680, 0.0005, 0.0008, 0.1698, 0.0041],
           [0.03220, 0.00050, 0.00500, 0.0837, 0.0615, 0.0514, 0.2231],
           [0.0366,  0.0004,  0.0001,  0.0733, 0.4509, 0.0036, 0.0036],
           [0.0096,  0.0176,  0.0014,  0.0086, 0.1216, 0.0177, 0.0068],
           [0.0068,  0.0102,  0.1011,  0.1012, 0.0120, 0.0728, 0.0479],
           [0.1147,  0.0021,  0.0002,  0.2157, 0.4744, 0.0102, 0.0017]]

    b_1 = [[0.000032, 0, 0, 0.000048, 0],
           [0, 0.308431, 0, 0, 0],
           [0, 0.000028, 0.000672, 0, 0.000028],
           [0, 0, 0.000340, 0.000097, 0],
           [0, 0.000200, 0.000223, 0.000006, 0.002337],
           [0, 0, 0.010446, 0, 0],
           [0, 0, 0, 0.506099, 0]]

    answer = viterbi_decode(os, ss, a_1, b_1, pie_1, pie_2)
    print(answer)

