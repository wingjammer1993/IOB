

def define_training_unk_words(word_frequency_count, sentence_sequence_word_list):
    sliced_dictionary = {k: v for k, v in word_frequency_count.items() if v <= 1}
    for index, word in enumerate(sentence_sequence_word_list):
        if word:
            if word in sliced_dictionary:
                sentence_sequence_word_list[index] = 'unk'
    return sentence_sequence_word_list


def define_extracted_unk_words(vocabulary_list, extracted_input_list):
    vocabulary = set(vocabulary_list)
    for sublist in extracted_input_list:
        for index, word in enumerate(sublist):
            if word not in vocabulary:
                sublist[index] = 'unk'
    return extracted_input_list


def extract_input_sentences_list(all_inputs):
    extracted_input = []
    observation_sequence = []
    for index, word in enumerate(all_inputs):
        if word != '<s>':
            if word != '<e>':
                observation_sequence.append(all_inputs[index])
            else:
                extracted_input.append(observation_sequence)
                observation_sequence = []

    return extracted_input


def extract_possible_tags_list(extracted_sentences, i_word_tag_pairs):
    extracted_possible_tags = []
    extracted_tag_sublist = []
    for sublist in extracted_sentences:
        for element in sublist:
            for tag in i_word_tag_pairs:
                if element == tag[1]:
                    if tag[0] not in extracted_tag_sublist:
                        extracted_tag_sublist.append(tag[0])
        extracted_possible_tags.append(extracted_tag_sublist)
        extracted_tag_sublist = []
    return extracted_possible_tags





