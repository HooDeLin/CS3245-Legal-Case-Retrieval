#!/usr/bin/python
import re
import os
import sys
import math
import getopt
import pickle
import psutil
import itertools
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import csv  # TODO: Either use this or pandas. Remove the unused one.
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
def is_stopword(token):
    return token in stopwords_set

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

def load_whole_dataset_csv(input_directory):
    df = pd.read_csv(input_directory)
    df = df.set_index("document_id", drop=False)
    df = df.drop_duplicates(("document_id", "content"), keep='last')    # TODO: Pick highest court. Currently picking the last one.
    df.sort_index()     # In case doc IDs are not sorted in increasing values
    return df

def get_docID_to_terms_mapping(df):
    """
    Processes the contents of corpus in df.content and returns a dictionary which
    maps docID to terms (consist of repeats).
    """
    docID_to_unigrams_dict = dict()    # Contains repeating words
    print("\tProcessing corpus...", flush=True) # TODO: Remove before submission.

    # TODO: Remove before submission.
    count = 0
    num_docs = len(df)

    for docID in df.index:
        raw_content = df.loc[docID, 'title'] + ' ' + df.loc[docID, 'content']   # TODO: Document decision to combine title and content

        content = re.sub(r'\s{2,}', ' ', raw_content)   # Efficiency purpose: Shorten string.
        content = remove_html_css_js(content)
        processed_terms_list = preprocess_string(content)

        docID_to_unigrams_dict[docID] = processed_terms_list  # Unigrams (may be repeated)

        # TODO: Remove before submission.
        count += 1
        if (count % 5== 0):
            print("\t\tProcessed {}/{} documents... (doc {})".format(count, num_docs, docID), flush=True)

    return docID_to_unigrams_dict

def get_citation_to_docID_maping(df, sorted_docIDs):
    """
    Processes the contents of corpus in df.content and returns a dictionary which
    maps citations to docIDs
    """
    citation_to_docID_dict = dict()
    for docID in sorted_docIDs:
        raw_content = df.loc[docID, 'title'] + ' ' + df.loc[docID, 'content']

        citation = get_citation(raw_content)
        if (citation != None):
            citation_to_docID_dict[citation] = docID
    return citation_to_docID_dict

def reverse_docID_to_terms_mapping(docID_to_unigrams_dict):
    """
    Parameters:
        A dictionary mapping docID to a list of its preprocessed terms
    Returns:
        A dictionary mapping each term to docIDs that contain it in ascending order
    """
    term_to_docIDs_dict = dict()

    for docID in sorted(docID_to_unigrams_dict):
        terms_set = set(docID_to_unigrams_dict[docID])
        for term in terms_set:
            if(term not in term_to_docIDs_dict):
                term_to_docIDs_dict[term] = []
            term_to_docIDs_dict[term].append(docID)

    return term_to_docIDs_dict

def build_unigram_postings(docID_to_unigrams_dict, min_df):
    """
    Build unigram postings (docID, normalized_tf-idf) given a dictionary that maps docID
    to terms in the document (including repeated words).
    """
    unigram_postings_dict = dict()  # Unigram postings are [docID, normalized tf-idf] pairs

    # TODO: Logging. Remove before submission
    count = 0

    for docID in sorted(docID_to_unigrams_dict):   # TODO: sorted() is used in many functions. Consider doing it only once
        terms_list = docID_to_unigrams_dict[docID]
        term_to_tf_dict = dict(Counter(terms_list))

        # Compute w_td and normalizing factor (magnitude of doc vector)
        accum_mag = 0   # Cumulative sum of squares of element doc_vec magnitude as normalizing factor
        for (term, tf) in term_to_tf_dict.items():
            w_td = log_tf(tf)
            accum_mag += w_td ** 2
        mag_doc_vec = math.sqrt(accum_mag)

        for (term, tf) in term_to_tf_dict.items():
            normalized_w_td = log_tf(tf) / mag_doc_vec
            if (term not in unigram_postings_dict):
                unigram_postings_dict[term] = list()
            unigram_postings_dict[term].append((docID, normalized_w_td))

    unigram_postings_dict = {term: postings for term, postings in unigram_postings_dict.items()
                            if len(postings) >= min_df}
    return unigram_postings_dict

def build_bigram_postings(docID_to_unigrams_dict, min_df):
    """
    Given a dictionary that maps docID to terms the docID contains (possibly with repeats),
    return a dictionary that maps bigrams to (docID) Boolean postings.
    """
    bigrams_postings_dict = dict()
    for docID in sorted(docID_to_unigrams_dict):
        processed_terms_list = docID_to_unigrams_dict[docID]
        for i in range(len(processed_terms_list) - 1):  # Bigrams
            bigram = " ".join(processed_terms_list[i:i+2])
            if (bigram not in bigrams_postings_dict):
                bigrams_postings_dict[bigram] = set()
            bigrams_postings_dict[bigram].add(docID)

    bigrams_postings_dict = {term: postings for term, postings in bigrams_postings_dict.items()
                                if len(postings) >= min_df}
    return bigrams_postings_dict

def build_trigram_postings(docID_to_unigrams_dict, min_df):
    """
    Given a dictionary that maps docID to terms the docID contains (possibly with repeats),
    return a dictionary that maps trigrams to (docID) Boolean postings.
    """
    trigrams_postings_dict = dict()
    for docID in sorted(docID_to_unigrams_dict):
        processed_terms_list = docID_to_unigrams_dict[docID]
        for i in range(len(processed_terms_list) - 2):  # Trigrams
            trigram = " ".join(processed_terms_list[i:i+3])
            if (trigram not in trigrams_postings_dict):
                trigrams_postings_dict[trigram] = set()
            trigrams_postings_dict[trigram].add(docID)

    trigrams_postings_dict = {term: postings for term, postings in trigrams_postings_dict.items()
                                if len(postings) >= min_df}
    return trigrams_postings_dict

def merge_itmd_index_postings(output_file_dictionary, output_file_postings, itmd_output_dir,
                              num_blocks, num_docs, min_unigram_df, min_multigram_df):
    """
    Merge intermediate block indices and postings, and save them as `output_file_dictionary` and `output_file_postings` respectively.
    """
    merged_index = Index()      # TODO: Currently assuming the entire index fit in memory. Reasonable?
    print("output file name: {}".format(output_file_postings))
    with open(output_file_postings, 'wb') as postings_fout:
        # Merge the index and postings across all intermediate files, term by term

        print("Accumulating all terms")     # TODO: Remove before submission
        sorted_terms = []
        for block_num in range(num_blocks):
            block_index = load_index(os.path.join(itmd_output_dir, "dictionary{}.txt".format(block_num)))
            sorted_terms.append([term for term in block_index])
        sorted_terms.sort()
        num_selected_terms = len(sorted_terms)    # TODO: Remove before submission

        print("No. of selected terms after filtering: {}".format(num_selected_terms))  # TODO: Remove before submission
        num_processed_terms = 0

        print("Merging intermediate index (without idf)...")  # TODO: Remove before submission.
        while(len(sorted_terms) != 0):  # TODO: Sorting is not actually strictly necessary. It helps in debugging though.
            # TODO: Remove before submission
            num_processed_terms += 1
            if (num_processed_terms%50 == 0):
                print("\tProcessing postings of {}...[{}/{}]".format(term, num_processed_terms, num_selected_terms), flush=True)

            accum_postings = []

            # Merge all blocks' postings of the current term into `accum_postings`
            for block_num in range(num_blocks):     # Note: num_blocks is the number of blocks where the counting starts from 1
                block_index = load_index(os.path.join(itmd_output_dir, "dictionary{}.txt".format(block_num)))
                block_postings_fin = open(os.path.join(itmd_output_dir, "postings{}.txt".format(block_num)), 'rb')

                block_postings = get_postings(term, block_index, block_postings_fin)
                accum_postings.extend(block_postings)

            offset = postings_fout.tell()
            postings_byte = pickle.dumps(accum_postings)
            postings_size = sys.getsizeof(postings_byte)
            postings_fout.write(postings_byte)

            merged_index.add_term_entry(term, offset, postings_size)

            sorted_terms = sorted_terms[1:]

    print("Saving 'dictionary.txt'...")  # TODO: Remove before submission.
    with open(output_file_dictionary, 'wb') as dictionary_file:
        pickle.dump(merged_index, dictionary_file)

    # TODO: Logging. Remove before submission
    print("Logging index to `log-index.txt`...", flush=True)
    with open(output_file_postings, 'rb') as postings_fin:
        with open('log-index.txt', 'w') as log_index_fout:
            for term in sorted(merged_index):
                postings = get_postings(term, merged_index, postings_fin)
                log_index_fout.write("'{}' --> {}\n".format(term, postings))

    return

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

    df = load_whole_dataset_csv(input_directory)
    num_docs = len(df)

    # with open(input_directory) as csvfile:
        # csv_reader = csv.reader(csvfile, delimiter=',')
        # for row in csv_reader:
        # pass

    citation_to_docID_dict = get_citation_to_docID_maping(df, df.index)
    print("Saving 'citation-docID.txt'...")     # TODO: Remove before submission
    with open('citation-docID.txt', 'wb') as citation_to_docID_file:
        pickle.dump(citation_to_docID_dict, citation_to_docID_file)

    # TODO: Naive logging. Remove before submission.
    log_citation_fout = open('log-docID-citation.txt', 'w')
    docID_to_citation_dict = dict()
    for citation, docID in citation_to_docID_dict.items():
        docID_to_citation_dict[docID] = citation
    for docID in df.index:
        if docID in docID_to_citation_dict:
            log_citation_fout.write("{} --> {}\n".format(docID, docID_to_citation_dict[docID]))
        else:
            log_citation_fout.write("{} --> [WARNING] Not found\n".format(docID))
    log_citation_fout.close()
    del docID_to_citation_dict

    # Free up RAM
    del df
    del citation_to_docID_dict

    # TODO: Consider using tf-idf for bigrams?
    min_unigram_df = 3      # TODO: But this causes problems when search for person's name
    max_unigram_df = math.ceil(0.6 * num_docs)     # TODO: Factor
    min_multigram_df = 3
    max_multigram_df = math.ceil(0.15 * num_docs)
    num_docs_per_block = 500
    block_num = 0

    itmd_output_dir = "temp"
    if (not os.path.exists(itmd_output_dir) and not os.path.isdir(itmd_output_dir)):
        os.mkdir("temp")    # To store intermediate SPIMI dictionary and postings
    else:   # TODO: Delete the intermediate index files when merging instead
        for filename in os.listdir(itmd_output_dir):
            filepath = os.path.join(itmd_output_dir, filename)
            os.remove(filepath)

    for df in pd.read_csv(input_directory, chunksize=num_docs_per_block):
        # TODO: Remove before submission if only used for logging
        print("Processing {}-th block...".format(block_num))

        df = df.set_index("document_id", drop=False)
        df = df.drop_duplicates(("document_id", "content"), keep='last')    # TODO: Pick highest court. Currently picking the last one.
        df.sort_index()     # In case doc IDs are not sorted in increasing values

        # Parse corpus: Preprocess contents, create postinsg and map terms to docID
        docID_to_unigrams_dict = get_docID_to_terms_mapping(df) # DS to facilitate tf calculation
        print("\tBuilding postings...")     # TODO: Remove befores submission
        min_block_df = 2
        unigram_postings_dict = build_unigram_postings(docID_to_unigrams_dict, min_block_df)
        bigram_postings_dict = build_bigram_postings(docID_to_unigrams_dict, min_block_df)
        trigram_postings_dict = build_trigram_postings(docID_to_unigrams_dict, min_block_df)
        del docID_to_unigrams_dict     # Free up RAM

        print("\tSaving 'postings{}.txt'...".format(block_num))  # TODO: Remove before submission.
        # Intermediate index maps terms to (offset, postings_byte_size) tuples
        # Postings are (docID, normalized w_td) tuples
        block_index = Index()

        """
        Structure of postings_file:
        Segment 1 - Unigram index
        Segment 2 - Bigram index
        Segment 3 - Trigram index
        """
        if (num_docs <= num_docs_per_block):
            block_num = ''  # TODO: Hacky way to avoid merging. Change if there's time
        with open(os.path.join(itmd_output_dir, 'postings{}.txt'.format(block_num)), 'wb') as postings_file:
            # TODO: 3 blocks of repeated code. Refactor as a function.
            for term in unigram_postings_dict:
                offset = postings_file.tell()
                postings_byte = pickle.dumps(unigram_postings_dict[term])
                postings_size = sys.getsizeof(postings_byte)

                block_index.add_unigram_entry(term, offset, postings_size)
                postings_file.write(postings_byte)

            del unigram_postings_dict

            for bigram in bigram_postings_dict:
                offset = postings_file.tell()
                postings_byte = pickle.dumps(bigram_postings_dict[bigram])
                postings_size = sys.getsizeof(postings_byte)

                block_index.add_bigram_entry(bigram, offset, postings_size)
                postings_file.write(postings_byte)

            del bigram_postings_dict

            for trigram in trigram_postings_dict:
                offset = postings_file.tell()
                postings_byte = pickle.dumps(trigram_postings_dict[trigram])
                postings_size = sys.getsizeof(postings_byte)

                block_index.add_trigram_entry(trigram, offset, postings_size)
                postings_file.write(postings_byte)

            del trigram_postings_dict

        print("\tSaving 'dictionary{}.txt'...".format(block_num))  # TODO: Remove before submission.
        with open(os.path.join(itmd_output_dir, 'dictionary{}.txt'.format(block_num)), 'wb') as dictionary_file:
            pickle.dump(block_index, dictionary_file)

        # TODO: Logging. Remove before submission
        with open(os.path.join(itmd_output_dir, 'postings{}.txt'.format(block_num)), 'rb') as postings_fin:
            with open(os.path.join(itmd_output_dir, 'log-index{}.txt'.format(block_num)), 'w') as log_index_fout:
                for term in sorted(block_index):
                    postings = get_postings(term, block_index, postings_fin)
                    log_index_fout.write("'{}' --> {}\n".format(term, postings))

        block_num += 1

    if (num_docs > num_docs_per_block):
        # Block num happens to be the number of blocks, where the counting starts at 1
        merge_itmd_index_postings(output_file_dictionary, output_file_postings,
                                  itmd_output_dir, block_num, num_docs,
                                  min_unigram_df, min_multigram_df)

#################
# For search.py #
#################

class Index:
    """
    Represents an index mapping:
        unigrams -> (offset, size, idf)
        bigrams -> (offset, size)
        trigrams -> (offset, size)
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

    @staticmethod
    def is_unigram(term):
        num_tokens = len(term.split())
        return num_tokens == 1

    @staticmethod
    def is_bigram(term):
        num_tokens = len(term.split())
        return num_tokens == 2

    @staticmethod
    def is_trigram(term):
        num_tokens = len(term.split())
        return num_tokens == 3

    def add_term_entry(self, term, offset, size):
        """Add a non-exsiting term to the index"""
        self.__dict[term] = (offset, size)
        return

    def add_unigram_entry(self, term, offset, size, idf=None):
        """Add a non-existing unigram to the index."""
        if (idf is not None):
            self.__dict[term] = (offset, size, idf)
        else:
            self.__dict[term] = (offset, size)
        return

    def add_bigram_entry(self, bigram, offset, size):
        """Add a non-existing bigram to the index."""
        self.__dict[bigram] = (offset, size)
        return

    def add_trigram_entry(self, trigram, offset, size):
        """Add a non-existing trigram to the index."""
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
    preprocess_string.lmtzr = WordNetLemmatizer()
    preprocess_string.stemmer = PorterStemmer()

    string = raw_string.casefold()
    string = re.sub(r'[^a-zA-Z\s]', '', string) # TODO: Remove punctuations and numbers. Good idea?
    tokens = word_tokenize(string)

    processed_tokens = []
    for token in tokens:
        if (is_stopword(token)
        and preprocess_string.lmtzr.lemmatize(preprocess_string.stemmer.stem(token))
        and not re.fullmatch(r'[a-zA-Z]{1,2}', token)):
            processed_tokens.append(token)

    return processed_tokens


##################################
# Procedural Program Starts Here #
##################################

if (__name__ == '__main__'):
    main()

    print("index.py finished running! :)")  # TODO: Remove before submission
