import concurrent.futures
import getopt
import time
import pickle
import re
import sys
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
from functools import reduce
from math import sqrt, log10
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

DEBUG_MODE = True
class Logger:
    def __init__(self, debug_mode=True):
        self._debug_mode = debug_mode
        self._start_time = 0
    
    def _set_start_time(self):
        self._start_time = time.time()

    def _log_end_time(self):
        print("Time used: {} seconds".format(time.time() - self._start_time))
    
    def log_start_loading_dataset(self):
        if self._debug_mode:
            print("Loading dataset into memory...")
            self._set_start_time()
    
    def log_end_loading_dataset(self, num_docs):
        if self._debug_mode:
            print("Finished loading dataset into memory...")
            print("Number of docs:{}".format(num_docs))
            self._log_end_time()
    
    def log_start_block_indexing(self):
        if self._debug_mode:
            print("Starting to indexing by blocks, this might take a while...")
            self._set_start_time()

    def log_end_block_indexing(self):
        print("FInished indexing all blocks")
        self._log_end_time()

    def log_finish_indexing_block(self, result):
        if self._debug_mode:
            print("Finished indexing block: {}".format(result))

logger = Logger(debug_mode=DEBUG_MODE)

def index_by_chunks(document_chunks):
    block_names = []
    logger.log_start_block_indexing()
    with concurrent.futures.ProcessPoolExecutor() as executor: # Put back the code first
        for result in executor.map(invert, list(range(len(document_chunks))), document_chunks):
            logger.log_finish_indexing_block(result)
            block_names.append(result)
    logger.log_end_block_indexing()
    return block_names

def merge_blocks(counter, num_docs, block_names):
    if len(block_names) == 1:
        return block_names[0]
    
    pairs = list(zip(block_names[::2], block_names[1::2]))
    block_number = list(range(counter, counter + len(pairs)))
    num_docs_list = [num_docs] * len(pairs)
    dict_a = list(map(lambda x: x[0][0], pairs))
    dict_b = list(map(lambda x: x[1][0], pairs))
    post_a = list(map(lambda x: x[0][1], pairs))
    post_b = list(map(lambda x: x[1][1], pairs))
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(merge_itmd_index_postings, block_number, num_docs_list, dict_a, post_a, dict_b, post_b):
            results.append(result)
    if len(block_names) % 2 == 1:
        results.append(block_names[len(block_names)-1])
    return merge_blocks(counter + len(results), num_docs, results)

def invert(block_number, document_chunk):
    docID_to_unigrams_dict = get_docID_to_terms_mapping(document_chunk)
    unigram_postings_dict = build_unigram_postings(docID_to_unigrams_dict, list(map(lambda x: x[0], document_chunk)))
    block_index = Index()
    posting_file_name = "postings{}.txt".format(block_number)
    dictionary_file_name = "dictionary{}.txt".format(block_number)
    postings_file = open(posting_file_name, 'wb')
    dictionary_file = open(dictionary_file_name, 'wb')
    for term in unigram_postings_dict:
        offset = postings_file.tell()
        postings_byte = pickle.dumps(unigram_postings_dict[term])
        postings_size = sys.getsizeof(postings_byte)
        block_index.add_term_entry(term, offset, postings_size)
        postings_file.write(postings_byte)
    del unigram_postings_dict
    postings_file.close()
    pickle.dump(block_index, dictionary_file)
    dictionary_file.close()
    return dictionary_file_name, posting_file_name

def merge_itmd_index_postings(block_number, num_docs, dict_a_name, post_a_name, dict_b_name, post_b_name):
    # Note: A must be before B, so that we don't have to sort doc_ids
    dict_a = load_index(dict_a_name)
    post_a = open(post_a_name, 'rb')
    dict_b = load_index(dict_b_name)
    post_b = open(post_b_name, 'rb')
    new_dict_name = "dictionary{}.txt".format(block_number)
    new_dict = Index()
    new_post_name = "posting{}.txt".format(block_number)
    new_post_fp = open(new_post_name, "wb")
    terms = [term_a for term_a in dict_a]
    terms.extend([term_b for term_b in dict_b])
    for term in terms:
        new_post = []
        if term in dict_a:
            new_post.extend(get_postings(term, dict_a, post_a))
        if term in dict_b:
            new_post.extend(get_postings(term, dict_b, post_b))
        offset = new_post_fp.tell()
        postings_byte = pickle.dumps(new_post)
        postings_size = sys.getsizeof(postings_byte)
        new_post_fp.write(postings_byte)
        new_dict.add_term_entry(term, offset, postings_size, term_idf=idf(len(new_post), num_docs))
    new_post_fp.close()
    pickle.dump(new_dict, open(new_dict_name,"wb"))
    return (new_dict_name, new_post_name)


def build_unigram_postings(docID_to_unigrams_dict, sorted_doc_ids):
    """
    Build unigram postings (docID, normalized_tf-idf) given a dictionary that maps docID
    to terms in the document (including repeated words).
    """
    unigram_postings_dict = dict()  # Unigram postings are [docID, normalized tf-idf] pairs

    for docID in sorted_doc_ids:
        terms_list = docID_to_unigrams_dict[docID]
        term_to_tf_dict = dict(Counter(terms_list))
        # Compute w_td and normalizing factor (magnitude of doc vector)
        mag_doc_vec = sqrt(reduce(lambda x, y: x + y**2, term_to_tf_dict.values(), 0)) # Cumulative sum of squares of element doc_vec magnitude as normalizing factor
        for term, tf in term_to_tf_dict.items():
            normalized_w_td = log_tf(tf) / mag_doc_vec
            if term not in unigram_postings_dict:
                unigram_postings_dict[term] = list()
            unigram_postings_dict[term].append((docID, normalized_w_td))
        return unigram_postings_dict

def get_docID_to_terms_mapping(id_content_tuples):
    docID_to_unigrams_dict = dict()    # Contains repeating words
    for id_content_tuple in id_content_tuples:
        (docID, raw_content) = id_content_tuple
        content = re.sub(r'\s{2,}', ' ', raw_content)
        content = remove_html_css_js(content)
        processed_terms_list = preprocess_string(content)
        docID_to_unigrams_dict[docID] = processed_terms_list
    return docID_to_unigrams_dict

stopwords_set = set(stopwords.words('english'))
def is_stopword(token):
    return token in stopwords_set

def log_tf(tf):
    if (tf == 0):
        return 0
    else:
        return 1 + log10(tf)

def idf(df, N):
    return log10(N/df)

def load_index(index_file):
    """
    Returns a Index object
    """
    return pickle.load(open(index_file, 'rb'))

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
        # Should we remove words that are small? Suggestion: Shouldn't, small words are mostly stopwords
        # which are mainly caught before this, only trigger regex to run once
        if not is_stopword(token):
            processed_tokens.append(preprocess_string.lmtzr.lemmatize(preprocess_string.stemmer.stem(token)))

    return processed_tokens

def remove_html_css_js(raw_string):
    soup = BeautifulSoup(raw_string, "lxml")
    return soup.body.getText()

def load_whole_dataset_csv(input_directory):
    df = pd.read_csv(input_directory)
    df = df.set_index("document_id", drop=False)
    df = df.drop_duplicates(("document_id", "content"), keep='last')
    df.sort_index()
    df['combined_content'] = df.apply(lambda row: row["content"] + ' ' + row["content"], axis=1)
    tuples = [tuple(x) for x in df[['document_id', 'combined_content']].values]
    del df
    return tuples

def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")

def parse_input_arguments():
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
    
    return (input_directory, output_file_dictionary, output_file_postings)

def main():
    (input_directory, output_file_dictionary, output_file_postings) = parse_input_arguments()
    logger.log_start_loading_dataset()
    id_content_tuples = load_whole_dataset_csv(input_directory)
    num_docs = len(id_content_tuples)
    logger.log_end_loading_dataset(num_docs)
    # ## TODO: Bring back citation

    num_docs_per_block = 1000
    document_chunks = [id_content_tuples[i * num_docs_per_block:(i + 1) * num_docs_per_block] for i in range((num_docs + num_docs_per_block - 1) // num_docs_per_block )]
    block_file_names = index_by_chunks(document_chunks)
    # To skip the file indexing
    # block_file_names = [('dictionary0.txt', 'postings0.txt'), ('dictionary1.txt', 'postings1.txt'), ('dictionary2.txt', 'postings2.txt'), ('dictionary3.txt', 'postings3.txt'), ('dictionary4.txt', 'postings4.txt'), ('dictionary5.txt', 'postings5.txt'), ('dictionary6.txt', 'postings6.txt'), ('dictionary7.txt', 'postings7.txt'), ('dictionary8.txt', 'postings8.txt'), ('dictionary9.txt', 'postings9.txt'), ('dictionary10.txt', 'postings10.txt'), ('dictionary11.txt', 'postings11.txt'), ('dictionary12.txt', 'postings12.txt'), ('dictionary13.txt', 'postings13.txt'), ('dictionary14.txt', 'postings14.txt'), ('dictionary15.txt', 'postings15.txt'), ('dictionary16.txt', 'postings16.txt'), ('dictionary17.txt', 'postings17.txt')]
    print(merge_blocks(len(block_file_names), num_docs, block_file_names))

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

    def __len__(self):
        return len(self.__dict)

    def add_term_entry(self, term, offset, size, term_idf=None):
        """Add a non-existing unigram to the index."""
        if (idf is not None):
            self.__dict[term] = (offset, size, term_idf)
        else:
            self.__dict[term] = (offset, size)
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

##################################
# Procedural Program Starts Here #
##################################

if __name__ == "__main__":
    main()

    print("index.py finished running! :)")  # TODO: Remove before submission