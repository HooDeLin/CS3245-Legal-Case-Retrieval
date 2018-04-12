#!/usr/bin/python
import re
import os
import sys
import math
import getopt
import pickle
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from bs4 import BeautifulSoup

def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")

#############
# Constants #
#############
IDX_POSTINGS_DOCID = 0
IDX_POSTINGS_TFIDF = 1
IDX_DICT_OFFSET = 0
IDX_DICT_SIZE = 1
IDX_DICT_IDF = 2

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

stopwords_set = set(stopwords.words('english'))
def remove_eng_stopwords(token_list):
    token_list = [token for token in token_list if token not in stopwords_set]
    return token_list

def lmtz_and_stem(token_list):
    lmtz_and_stem.lmtzr = WordNetLemmatizer()
    lmtz_and_stem.stemmer = PorterStemmer()
    token_list = [lmtz_and_stem.lmtzr.lemmatize(lmtz_and_stem.stemmer.stem(token)) for token in token_list]
    return token_list

def log_tf(tf):
    if (tf == 0):
        return 0
    else:
        return 1 + math.log10(tf)

def idf(df, N):
    return math.log10(N/df)

def get_citation(raw_string):
    """
    Returns the neutral citation of a law report's content (string).
    Returns `None` if no citation is found.
    """
    get_citation.re = r'\[\d+\] (\d+ )?[A-Z](\.*[A-Z]+)* \d+'
    match_obj = re.search(get_citation.re, raw_string[:200])  # TODO: Find citation with first 200 chars only. Is it a good decision?

    if (match_obj == None):
        return None

    return match_obj.group(0)

def remove_html_css_js(raw_string):
    soup = BeautifulSoup(raw_string, "lxml")
    return soup.body.getText()

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

    postings_dict = dict()  # Postings are [docID, normalized tf-idf] pairs
    term_to_idf_dict = dict()
    citation_to_docID_dict = dict()
    num_docs = len(df)

    # TODO: Refactor this section into a func
    # First parse of collection -- accum docIDs for each term to compute idf
    print("Computing idf's...") # TODO: Remove before submission.
    docID_to_terms_list_dict = dict()
    term_to_docIDs_dict = dict()    # Temporary DS
    count = 0   # TODO: Remove before submission.
    for docID in df.index:  # Note that df.index is already sorted
        raw_content = df.loc[docID, 'title'] + ' ' + df.loc[docID, 'content']   # TODO: Currently combining title with content

        citation = get_citation(raw_content)
        # TODO: Remove. Logging
        if (citation != None):
            citation_to_docID_dict[citation] = docID

        raw_content = remove_html_css_js(raw_content)
        processed_terms_list = preprocess_string(raw_content)

        docID_to_terms_list_dict[docID] = []    # 'terms' here includes unigrams, bigrams and trigrams
        for i in range(len(processed_terms_list)):
            docID_to_terms_list_dict[docID].append(processed_terms_list[i])
        for i in range(len(processed_terms_list) - 1):
            docID_to_terms_list_dict[docID].append(" ".join(processed_terms_list[i:i+2]))
        for i in range(len(processed_terms_list) - 2):
            docID_to_terms_list_dict[docID].append(" ".join(processed_terms_list[i:i+3]))

        unique_terms_set = set(docID_to_terms_list_dict[docID])
        for term in unique_terms_set:
            if term not in term_to_docIDs_dict:
                term_to_docIDs_dict[term] = []
            term_to_docIDs_dict[term].append(docID)

        # TODO: Remove before submission.
        count += 1
        #if (count % 100 == 0):
        print("\tProcessed {}/{} documents... (doc {})".format(count, num_docs, docID))

    for term in term_to_docIDs_dict:
        term_to_idf_dict[term] = idf(len(term_to_docIDs_dict[term]), num_docs)
    del term_to_docIDs_dict

    print("Building postings...")   # TODO: Remove before submission.
    # Second parse of collection to build postings
    count = 0   # TODO: Remove before submission.
    for docID in df.index:  # Note that df.index is already sorted
        terms_list = docID_to_terms_list_dict[docID]    # 'terms' here include unigrams, bigrams and trigrams
        term_to_tf_dict = dict(Counter(terms_list))
        term_to_w_td_dict = dict()

        # Compute w_td and normalizing factor (magnitude of doc vector)
        accum_mag = 0   # Cumulative sum of squares of element doc_vec magnitude as normalizing factor
        for (term, tf) in term_to_tf_dict.items():
            w_td = log_tf(tf) + term_to_idf_dict[term]
            term_to_w_td_dict[term] = w_td
            accum_mag += w_td ** 2
        mag_doc_vec = math.sqrt(accum_mag)

        for (term, w_td) in term_to_w_td_dict.items():
            normalized_w_td = w_td / mag_doc_vec
            if (term not in postings_dict):
                postings_dict[term] = list()
            postings_dict[term].append((docID, normalized_w_td))

        # TODO: Remove before submission.
        count += 1
        if (count % 50 == 0) or (count == num_docs):
            print("\tBuilt postings for {}/{} documents...".format(count, num_docs))

    print("Saving 'dictionary.txt','postings.txt' and 'citation-docID.txt'...")  # TODO: Remove before submission.
    # Save to 'dictionary.txt' and 'postings.txt'
    # Dictionary maps terms to (offset, postings_byte_size, idf) tuples
    # Postings are (docID, normalized w_td) tuples
    dictionary_in_mem = dict()

    # TODO: Naive logging. Remove before submission.
    log_dictionary_fout = open('log-dictionary.txt', 'w')
    log_postings_fout = open('log-postings.txt', 'w')
    log_citation_fout = open('log-docID-citation.txt', 'w')

    with open(output_file_postings, 'wb') as postings_file:
        for term in sorted(postings_dict):
            offset = postings_file.tell()
            postings_byte = pickle.dumps(postings_dict[term])
            postings_size = sys.getsizeof(postings_byte)

            dictionary_in_mem[term] = (offset, postings_size, term_to_idf_dict[term])
            postings_file.write(postings_byte)

            # TODO: Naive logging. Remove before submission.
            log_dictionary_fout.write("'{}' --> {}, {}\n".format(term, offset, term_to_idf_dict[term]))
            log_postings_fout.write("'{}' --> {}\n".format(term, repr(postings_dict[term])))

    with open(output_file_dictionary, 'wb') as dictionary_file:
        pickle.dump(dictionary_in_mem, dictionary_file)

    with open('citation-docID.txt', 'wb') as citation_to_docID_file:
        pickle.dump(citation_to_docID_dict, citation_to_docID_file)

    # TODO: Naive logging. Remove before submission
    docID_to_citation_dict = dict()
    for citation, docID in citation_to_docID_dict.items():
        docID_to_citation_dict[docID] = citation

    for docID in df.index:
        if docID in docID_to_citation_dict:
            log_citation_fout.write("{} --> {}\n".format(docID, docID_to_citation_dict[docID]))
        else:
            log_citation_fout.write("{} --> [WARNING] Not found\n".format(docID))

    # TODO: Naive logging. Remove before submission.
    log_dictionary_fout.close()
    log_postings_fout.close()
    log_citation_fout.close()

#################
# For search.py #
#################

def load_dictionary(dictionary_file):
    """
    Returns a dictionary mapping
    "term" -> (offset, size, idf)
    """
    return pickle.load(open(dictionary_file, 'rb'))

def load_citation_to_docID_dict():
    """
    Returns a dictionary mapping
    citation -> docID
    """
    return pickle.load(open('citation-docID.txt', 'rb'))

def get_postings(term, dictionary, postings_reader):
    """
    Parameters
        dictionary: A dictionary mapping 'terms' -> (offset, size, idf)
        postings_reader: A postings file object with methods seek() and readline()
    Returns
        A list of (docID, normalized tf-idf) postings
    """
    
    assert(type(term) == str)
    if (term not in dictionary):
        return []

    offset = dictionary[term][IDX_DICT_OFFSET]
    postings_size = dictionary[term][IDX_DICT_SIZE]

    postings_reader.seek(offset, 0)
    postings_byte = postings_reader.read(postings_size)
    postings = pickle.loads(postings_byte)
    return postings

def get_idf(term, dictionary):
    """
    Parameters
        term: The term which idf is to be found
        dictionary: A dictionary mapping 'terms' -> (offset, size, idf)
    Returns
        idf
    """
    return dictionary[term][IDX_DICT_IDF]

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
    string = remove_punctuations(string)
    string = re.sub(r'[0-9]', '', string) # TODO: Rethink whether removing numbers is a good idea or not
    tokens = word_tokenize(string)
    tokens = remove_eng_stopwords(tokens)
    tokens = lmtz_and_stem(tokens)
    return tokens


##################################
# Procedural Program Starts Here #
##################################
    
if (__name__ == '__main__'):
    main()

    print("index.py finished running! :)")  # TODO: Remove before submission
