import nltk
import CountFrequency
import TPM


# This method is generating bigram tuples from the parsed file, using nltk

def get_epm_bigrams(sequence_1, sequence_2):
    bigram = list(zip(sequence_1, sequence_2))
    return bigram


# This method is taking the bigram list, unigram counts, and the filename1&2.
# For every tag in the filename (tag list), it is finding out how many times that tag is followed by a word from vocab.
# It then arranges this data in a tag x word matrix
# It returns a list of dictionaries, each dictionary corresponds to a column of emission probability matrix
# dictionaries are ordered according to tag list

def get_tag_bigram_probability(filename_tag, vocabulary, bigram_list, unigram_count):
    word_tag_dictionary_list = {}
    with open(filename_tag, encoding="utf8") as file_obj:
        content = file_obj.readlines()
        for tag_name_row in content:
            if tag_name_row:
                tag_successors = [b for (a, b) in bigram_list if a == tag_name_row.strip()]
                freq_dist = nltk.FreqDist(tag_successors)
                unigram_tag_count = unigram_count.get(tag_name_row.strip())
                word_tag_dictionary = {}
                for word_name_column in vocabulary:
                    if word_name_column:
                        tag_bigram_count = freq_dist[word_name_column.strip()]
                        if unigram_tag_count:
                            word_tag_dictionary[word_name_column.strip()] = tag_bigram_count/unigram_tag_count
                        else:
                            word_tag_dictionary[word_name_column.strip()] = 0
                word_tag_dictionary_list[tag_name_row.strip()] = word_tag_dictionary

    return word_tag_dictionary_list


# This method calls all other methods

def get_emission_probability_matrix(tags_file, vocabulary, word_tag_pairs, tag_frequency_count):
    wtag_bigram_count_dictionary = get_tag_bigram_probability(tags_file, vocabulary, word_tag_pairs, tag_frequency_count)
    return wtag_bigram_count_dictionary
