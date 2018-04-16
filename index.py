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

    sorted_docIDs = df.index    # To facilitate iterating docIDs in sorted order
    num_docs = len(df)

    unigram_postings_dict = dict()  # Unigram postings are [docID, normalized tf-idf] pairs
    bigram_postings_dict = dict()    # Bigram postings are Boolean postings, just unique and non-decreasing set of docIDs
    trigram_postings_dict = dict()   # Trigram postings are Boolean postings, just unique and non-decreasing set of docIDs

    term_to_idf_dict = dict()   # Intermediate DS for creating index
    citation_to_docID_dict = dict()

    # TODO: Refactor this section into a func
    # First parse of collection -- extract citations & accum docIDs for each term to compute idf
    print("Processing corpus...", flush=True) # TODO: Remove before submission.
    docID_to_terms_list_dict = dict()
    term_to_docIDs_dict = dict()    # Temporary DS
    count = 0   # TODO: Remove before submission.
    for docID in sorted_docIDs:
        raw_content = df.loc[docID, 'title'] + ' ' + df.loc[docID, 'content']   # TODO: Combine title with content. Good idea?

        citation = get_citation(raw_content)
        # TODO: Remove. Logging
        if (citation != None):
            citation_to_docID_dict[citation] = docID

        content = re.sub(r'\s{2,}', ' ', raw_content)   # Efficiency purpose: Shorten string.
        content = remove_html_css_js(content)
        processed_terms_list = preprocess_string(content)

        docID_to_terms_list_dict[docID] = processed_terms_list  # Unigrams   
        for i in range(len(processed_terms_list) - 1):  # Bigrams
            bigram = " ".join(processed_terms_list[i:i+2])
            if (bigram not in bigram_postings_dict):
                bigram_postings_dict[bigram] = set()
            bigram_postings_dict[bigram].add(docID)
        for i in range(len(processed_terms_list) - 2):  # Trigrams
            trigram = " ".join(processed_terms_list[i:i+3])
            if (trigram not in trigram_postings_dict):
                trigram_postings_dict[trigram] = set()
            trigram_postings_dict[trigram].add(docID)

        unique_terms_set = set(docID_to_terms_list_dict[docID])
        for term in unique_terms_set:
            if term not in term_to_docIDs_dict:
                term_to_docIDs_dict[term] = []
            term_to_docIDs_dict[term].append(docID)

        # TODO: Remove before submission.
        count += 1
        #if (count % 100 == 0):
        print("\tProcessed {}/{} documents... (doc {})".format(count, num_docs, docID), flush=True)

    del df      # Free up RAM. df is large

    # TODO: Remove before submission
    print("Saving 'citation-docID.txt'...")
    with open('citation-docID.txt', 'wb') as citation_to_docID_file:
        pickle.dump(citation_to_docID_dict, citation_to_docID_file)
    # TODO: Naive logging. Remove before submission
    log_citation_fout = open('log-docID-citation.txt', 'w')
    docID_to_citation_dict = dict()
    for citation, docID in citation_to_docID_dict.items():
        docID_to_citation_dict[docID] = citation

    for docID in sorted_docIDs:
        if docID in docID_to_citation_dict:
            log_citation_fout.write("{} --> {}\n".format(docID, docID_to_citation_dict[docID]))
        else:
            log_citation_fout.write("{} --> [WARNING] Not found\n".format(docID))
    log_citation_fout.close()
    del docID_to_citation_dict

    del citation_to_docID_dict  # Free up RAM

    print("Computing idf's...") # TODO: Remove before submission.
    for term in term_to_docIDs_dict:
        term_to_idf_dict[term] = idf(len(term_to_docIDs_dict[term]), num_docs)
    del term_to_docIDs_dict

    print("Building postings...")   # TODO: Remove before submission.
    # Second parse of collection to build postings
    count = 0   # TODO: Remove before submission.
    for docID in sorted_docIDs:
        terms_list = docID_to_terms_list_dict[docID]
        term_to_tf_dict = dict(Counter(terms_list))
        term_to_w_td_dict = dict()

        # Compute w_td and normalizing factor (magnitude of doc vector)
        accum_mag = 0   # Cumulative sum of squares of element doc_vec magnitude as normalizing factor
        for (term, tf) in term_to_tf_dict.items():
            w_td = log_tf(tf) * term_to_idf_dict[term]
            term_to_w_td_dict[term] = w_td
            accum_mag += w_td ** 2
        mag_doc_vec = math.sqrt(accum_mag)

        for (term, w_td) in term_to_w_td_dict.items():
            normalized_w_td = w_td / mag_doc_vec
            if (term not in unigram_postings_dict):
                unigram_postings_dict[term] = list()
            unigram_postings_dict[term].append((docID, normalized_w_td))

        # TODO: Remove before submission.
        count += 1
        if (count % 50 == 0) or (count == num_docs):
            print("\tBuilt postings for {}/{} documents...".format(count, num_docs), flush=True)

    print("Saving 'dictionary.txt','postings.txt'...")  # TODO: Remove before submission.
    # Save to 'dictionary.txt' and 'postings.txt'
    # Index maps terms to (offset, postings_byte_size, idf) tuples
    # Postings are (docID, normalized w_td) tuples
    index = Index()

    # TODO: Naive logging. Remove before submission.
    log_index_fout = open('log-index.txt', 'w')
    log_postings_fout = open('log-postings.txt', 'w')

    """
    Structure of postings_file:
    Segment 1 - Unigram index
    Segment 2 - Bigram index
    Segment 3 - Trigram index
    Segment 4 - Citations-to-docIDs mapping (TODO: maybe?)
    """
    with open(output_file_postings, 'wb') as postings_file:
        # TODO: 3 blocks of repeated code. Refactor as a function.
        for term in sorted(unigram_postings_dict):
            offset = postings_file.tell()
            postings_byte = pickle.dumps(unigram_postings_dict[term])
            postings_size = sys.getsizeof(postings_byte)

            index.add_unigram_entry(term, offset, postings_size, term_to_idf_dict[term])
            postings_file.write(postings_byte)

            # TODO: Naive logging. Remove before submission.
            log_index_fout.write("'{}' --> {}, {}, {}\n".format(term, offset, postings_size, term_to_idf_dict[term]))
            log_postings_fout.write("'{}' --> {}\n".format(term, unigram_postings_dict[term]))
        del unigram_postings_dict

        for bigram in sorted(bigram_postings_dict):
            offset = postings_file.tell()
            postings_byte = pickle.dumps(bigram_postings_dict[bigram])
            postings_size = sys.getsizeof(postings_byte)

            index.add_bigram_entry(bigram, offset, postings_size)
            postings_file.write(postings_byte)

            # TODO: Naive logging. Remove before submission.
            log_index_fout.write("'{}' --> {}\n".format(bigram, offset))
            log_postings_fout.write("'{}' --> {}\n".format(bigram, bigram_postings_dict[bigram]))
        del bigram_postings_dict

        for trigram in sorted(trigram_postings_dict):
            offset = postings_file.tell()
            postings_byte = pickle.dumps(trigram_postings_dict[trigram])
            postings_size = sys.getsizeof(postings_byte)

            index.add_trigram_entry(trigram, offset, postings_size)
            postings_file.write(postings_byte)

            # TODO: Naive logging. Remove before submission.
            log_index_fout.write("'{}' --> {}\n".format(trigram, offset))
            log_postings_fout.write("'{}' --> {}\n".format(trigram, trigram_postings_dict[trigram]))
        del trigram_postings_dict
    del term_to_idf_dict

    with open(output_file_dictionary, 'wb') as dictionary_file:
        pickle.dump(index, dictionary_file)
    del index

    # TODO: Naive logging. Remove before submission.
    log_index_fout.close()
    log_postings_fout.close()

#################
# For search.py #
#################

class Index:
    """
    Represents an index mapping:
        terms -> (offset, size, idf)
    """
    __offset_idx = 0
    __size_idx = 1
    __idf_idx = 2
    
    def __init__(self):
        self.__dict = dict()

    def __str__(self):
        return self.__dict.__str__()

    def __iter__(self):
        return self.__dict.__iter__()

    def __contains__(self, key):
        return key in self.__dict

    def add_unigram_entry(self, term, offset, size, idf):
        """Add a non-existing term to the index."""
        self.__dict[term] = (offset, size, idf)
        return

    def add_bigram_entry(self, bigram, offset, size):
        """Add a non-existing term to the index."""
        self.__dict[bigram] = (offset, size)
        return

    def add_trigram_entry(self, trigram, offset, size):
        """Add a non-existing term to the index."""
        self.__dict[trigram] = (offset, size)
        return

    def get_postings_offset(self, term):
        return self.__dict[term][Index.__offset_idx]

    def get_postings_size(self, term):
        """Get size of postings in bytes"""
        return self.__dict[term][Index.__size_idx]

    def get_idf(self, term):
        if term not in self.__dict:
            return 0
        else:
            return self.__dict[term][Index.__idf_idx]

def load_index(index_file):
    """
    Returns a Index object
    """
    return pickle.load(open(index_file, 'rb'))

def load_citation_to_docID_dict():
    """
    Returns a dictionary mapping
    citation -> docID
    """
    return pickle.load(open('citation-docID.txt', 'rb'))

def get_postings(term, index, postings_reader):
    """
    Parameters
        Index object from index.py
        postings_reader: A postings file object with methods seek() and readline()
    Returns
        A list of (docID, normalized tf-idf) postings
    """
    
    assert(type(term) == str)
    if (term not in index):
        return []

    offset = index.get_postings_offset(term)
    postings_size = index.get_postings_size(term)

    postings_reader.seek(offset, 0)
    postings_byte = postings_reader.read(postings_size)
    postings = pickle.loads(postings_byte)
    return postings

def preprocess_string(raw_string):
    """
    Preprocess raw_string and returns a list of processed dictionary terms.

    Preprocessing order:
    - case-folding
    - remove punctuations and numbers
    - tokenization
    - remove english stopwords
    - lemmatization
    - stemming
    - remove too short words (1-2 chars long)
    """
    string = raw_string.casefold()
    string = re.sub(r'[^a-zA-Z\s]', '', string) # TODO: Remove punctuations and numbers. Good idea?
    tokens = word_tokenize(string)
    tokens = remove_eng_stopwords(tokens)
    tokens = lmtz_and_stem(tokens)
    tokens = [token for token in tokens if not re.fullmatch(r'[a-zA-Z]{1,2}', token)]  # TODO: Remove tokens that are too short (1 to 2 chars). Good idea?
    return tokens


##################################
# Procedural Program Starts Here #
##################################
    
if (__name__ == '__main__'):
    main()

    print("index.py finished running! :)")  # TODO: Remove before submission
