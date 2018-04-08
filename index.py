#!/usr/bin/python
import re
import os
import sys
import math
import getopt
import pickle
import shelve
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")

####################
# Helper Functions #
####################

def remove_punctuations(string):
    """
    Removes punctuations in the following way:
    1. Removes all '-', "'" or '.' that appears before alphabets, intention is to convert
    phrases like "mother-in-law" to "motherinlaw", "doesn't" to "doesnt", "u.s.a" to "usa",
    etc.
    2. Removes all ',' within digits that indicates thousands,
    e.g. "1,000,000" -> "1000000"
    3. Replaces all punctuations with a space.
    """
    modified_string = re.sub(r"(\w)['\-.](\w)", r'\1\2', string) # Punctuations within strings
    modified_string = re.sub(r",(\d)", r'\1', modified_string)  # Commas between digits
    modified_string = re.sub(r'[^a-zA-Z0-9\s]', ' ', modified_string)
    return modified_string

def lemmatize(token_list):
    lmtzr = WordNetLemmatizer()
    token_list = [lmtzr.lemmatize(token) for token in token_list]
    return token_list

stopwords_set = set(stopwords.words('english'))
def remove_eng_stopwords(token_list):
    token_list = [token for token in token_list if token not in stopwords_set]
    return token_list

def stem(token_list):
    stemmer = PorterStemmer()
    token_list = [stemmer.stem(token) for token in token_list]
    return token_list
    
# TODO: This will be a large function. Can consider refactoring later.
def preprocess_string(raw_string):
    """
    Preprocess raw_string and returns a list of processed dictionary terms.

    Preprocessing order:
    - case-folding
    - remove numbers (disabled)
    - tokenization
    - remove english stopwords
    - lemmatization
    - stemming
    """
    string = raw_string.casefold()
    # string = re.sub(r'[0-9]', '', string) # Uncomment to remove numeric characters
    string = remove_punctuations(string)
    tokens = word_tokenize(string)
    tokens = remove_eng_stopwords(tokens)
    tokens = lemmatize(tokens)
    tokens = stem(tokens)
    return tokens

def log_tf(tf):
    if (tf == 0):
        return 0
    else:
        return 1 + math.log10(tf)

def idf(df, N):
    return math.log10(N/df)

def main():
    # Command line inputs
    input_directory = output_file_dictionary = output_file_postings = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
    except getopt.GetoptError as err:
        usage()
        sys.exit(2)
        
    for o, a in opts:
        if o == '-i': # input directory
            input_directory = a
        elif o == '-d': # dictionary file
            output_file_dictionary = a
        elif o == '-p': # postings file
            output_file_postings = a
        else:
            assert False, "unhandled option"
            
    if input_directory == None or output_file_postings == None or output_file_dictionary == None:
        usage()
        sys.exit(2)

    print("Reading and processing 'dataset.csv'...")   # TODO: Remove before submission.
    df = pd.read_csv(input_directory)
    df = df.set_index("document_id", drop=False)
    df = df.drop_duplicates(("document_id", "content"), keep='last')    # TODO: Pick highest court. Currently picking the last one.
    df.sort_index()     # In case doc IDs are not sorted in increasing values

    # Nomenclature of variables: name_dtype

    # TODO: Create 2gram and 3 gram postings
    # Postings are [docID, normalized tf-idf] pairs
    postings_1gram_dict = dict()
    all_docIDs_list = []
    term_to_idf_dict = dict()

    # Constants
    IDX_POSTINGS_DOCID = 0
    IDX_POSTINGS_TFIDF = 1

    # TODO: Extract neutral citations

    # TODO: Refactor this section into a func
    # First parse of collection -- accum docIDs for each term to compute idf
    term_to_docIDs_dict = dict()    # Temporary DS
    for docID in df.index:
        all_docIDs_list.append(docID)

        raw_content = df.loc[docID, 'content']
        
        unique_terms = set(preprocess_string(raw_content))
        for term in unique_terms:
            if term not in term_to_docIDs_dict:
                term_to_docIDs_dict[term] = [docID]
            else:
                term_to_docIDs_dict[term].append(docID)

    num_docs = len(all_docIDs_list)
    for term in term_to_docIDs_dict:
        term_to_idf_dict[term] = idf(len(term_to_docIDs_dict[term]), num_docs)
    del term_to_docIDs_dict

    print("Building postings...")   # TODO: Remove before submission.
    # Second parse of collection to build postings
    for docID in df.index:  # Note that df.index is already sorted
        raw_content = df.loc[docID, 'content']
        
        terms_list = preprocess_string(raw_content)
        term_to_tf_dict = dict(Counter(terms_list))
        term_to_w_td_dict = dict()

        # Compute normalizing factor (magnitude of doc vector)
        accum_mag = 0   # Cumulative sum of squares of element doc_vec magnitude as normalizing factor
        for (term, tf) in term_to_tf_dict.items():
            w_td = log_tf(tf) + term_to_idf_dict[term]
            accum_mag += w_td ** 2
        mag_doc_vec = math.sqrt(accum_mag)

        for (term, w_td) in term_to_w_td_dict.items():
            normalized_w_td = w_td / mag_doc_vec
            if (term not in postings_1gram_dict):
                postings_1gram_dict[term] = list()
            postings_1gram_dict[term].append([docID, normalized_w_td])

    print("Saving 'dictionary.txt' and 'postings.txt'...")  # TODO: Remove before submission.
    # Save to 'dictionary.txt' and 'postings.txt'
    # Dictionary maps terms to (offset, idf) tuples
    # Postings are (docID, normalized w_td) tuples
    dictionary_shelf = shelve.open(output_file_dictionary)

    # TODO: Naive logging. Remove before submission.
    log_dictionary_fout = open('log-dictionary.txt', 'w')
    log_postings_fout = open('log-postings.txt', 'w')

    with open(output_file_postings, 'wb') as postings_file:
        for term in sorted(postings_1gram_dict):
            offset = postings_file.tell()
            dictionary_shelf[term] = (offset, term_to_idf_dict[term])
            pickle.dump(postings_1gram_dict[term], postings_file)

            # TODO: Naive logging. Remove before submission.
            log_dictionary_fout.write("'{}' --> {}, {}".format(term, offset, term_to_idf_dict[term]))
            log_postings_fout.write("'{}' --> {}".format(term, repr(postings_1gram_dict[term])))

    dictionary_shelf.close()

    # TODO: Naive logging. Remove before submission.
    log_dictionary_fout.close()
    log_postings_fout.close()
    
    # TODO: Create retrieval methods for De Lin

##################################
# Procedural Program Starts Here #
##################################
if (__name__ == '__main__'):
    main()

    print("index.py finished running! :)")  # TODO: Remove before submission
    
