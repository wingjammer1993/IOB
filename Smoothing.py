import nltk


def get_addk_tag_bigram_probability(filename, bigram_list, unigram_count):
    tag_dictionary_list = {}
    with open(filename, encoding="utf8") as file_obj:
        content = file_obj.readlines()
        for tag_name_row in content:
            if tag_name_row:
                tag_successors = [b for (a, b) in bigram_list if a == tag_name_row.strip()]
                freq_dist = nltk.FreqDist(tag_successors)
                unigram_tag_count = unigram_count.get(tag_name_row.strip())
                with open(filename, encoding="utf8") as re_file_obj:
                    tag_tag_dictionary = {}
                    for tag_name_column in re_file_obj.readlines():
                        if tag_name_column:
                            tag_bigram_count = freq_dist[tag_name_column.strip()]
                            if unigram_tag_count:
                                if tag_bigram_count:
                                    tag_tag_dictionary[tag_name_column.strip()] = (tag_bigram_count+0.001)/(unigram_tag_count+0.046)
                                else:
                                    tag_tag_dictionary[tag_name_column.strip()] = 0.001 / (unigram_tag_count + 0.046)
                            else:
                                tag_tag_dictionary[tag_name_column.strip()] = 0
                    tag_dictionary_list[tag_name_row.strip()] = tag_tag_dictionary

    return tag_dictionary_list


def get_backoff_tag_bigram_probability(filename, bigram_list, unigram_count):
    tag_dictionary_list = {}
    with open(filename, encoding="utf8") as file_obj:
        content = file_obj.readlines()
        for tag_name_row in content:
            if tag_name_row:
                tag_successors = [b for (a, b) in bigram_list if a == tag_name_row.strip()]
                freq_dist = nltk.FreqDist(tag_successors)
                total_count = sum(unigram_count.values())
                unigram_tag_count = unigram_count.get(tag_name_row.strip())
                with open(filename, encoding="utf8") as re_file_obj:
                    tag_tag_dictionary = {}
                    for tag_name_column in re_file_obj.readlines():
                        if tag_name_column:
                            tag_bigram_count = freq_dist[tag_name_column.strip()]
                            if unigram_tag_count:
                                if tag_bigram_count:
                                    tag_tag_dictionary[tag_name_column.strip()] = tag_bigram_count/unigram_tag_count
                                else:
                                    unigram_backoff_tag_count = unigram_count.get(tag_name_column.strip())
                                    if unigram_backoff_tag_count:
                                        tag_tag_dictionary[tag_name_column.strip()] = unigram_backoff_tag_count/total_count
                                    else:
                                        tag_tag_dictionary[tag_name_column.strip()] = 0
                            else:
                                tag_tag_dictionary[tag_name_column.strip()] = 0
                    tag_dictionary_list[tag_name_row.strip()] = tag_tag_dictionary

    return tag_dictionary_list


def get_backoff_smoothed_tpm(tags_file, tag_tag_pairs, tag_frequency_count):
    tag_bigram_count_dictionary = get_backoff_tag_bigram_probability(tags_file, tag_tag_pairs, tag_frequency_count)
    return tag_bigram_count_dictionary


def get_add_k_smoothed_tpm(tags_file, tag_tag_pairs, tag_frequency_count):
    tag_bigram_count_dictionary = get_addk_tag_bigram_probability(tags_file, tag_tag_pairs, tag_frequency_count)
    return tag_bigram_count_dictionary


