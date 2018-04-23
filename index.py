import concurrent.futures
import csv
import getopt
import time
import os
import pickle
import re
import sys
# from autocorrect import spell
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
        print("Finished indexing all blocks")
        self._log_end_time()

    def log_finish_indexing_block(self, result):
        if self._debug_mode:
            print("Finished indexing block: {}".format(result))

    def log_start_merge_blocks(self):
        if self._debug_mode:
            print("Starting to merge blocks...")
            self._set_start_time()

    def log_end_merge_blocks(self):
        if self._debug_mode:
            print("Finished merging all blocks")
            self._log_end_time()

    def log_start_calculating_idf(self):
        if self._debug_mode:
            print("Start calculating idf")
            self._set_start_time()

    def log_end_calculating_idf(self):
        if self._debug_mode:
            print("Finished calculating idf")
            self._log_end_time()

logger = Logger(debug_mode=DEBUG_MODE)

class Court:
    hierarchy = {
        "UK Supreme Court": 1,
        "UK House of Lords": 1,
        "UK Court of Appeal": 0.8,
        "UK High Court": 0.6,
        "UK Crown Court": 0.4,
        "UK Military Court": 0.4,

        "High Court of Australia": 1,
        "Federal Court of Australia": 0.8,
        "NSW Court of Criminal Appeal": 0.8,
        "NSW Court of Appeal": 0.8,
        "NSW Supreme Court": 0.6,
        "NSW District Court": 0.4,
        "NSW Industrial Court": 0.4,
        "NSW Administrative Decisions Tribunal (Trial)": 0.2,
        "NSW Children's Court": 0.2,
        "NSW Civil and Administrative Tribunal": 0.2,
        "NSW Industrial Relations Commission": 0.2,
        "NSW Land and Environment Court": 0.2,
        "NSW Local Court": 0.2,
        "NSW Medical Tribunal": 0.2,
        "Industrial Relations Court of Australia": 0.2,

        "SG High Court": 1,
        "SG Court of Appeal": 1,
        "SG District Court": 0.4,
        "SG Magistrates' Court": 0.4,
        "SG Family Court": 0.2,
        "SG Privy Council": 0.2,
        "Singapore International Commercial Court": 0.2,

        "HK Court of First Instance": 1,
        "HK High Court": 0.8,

        "CA Supreme Court": 1
    }

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
    dict_a = list(map(lambda x: x[0][0], pairs))
    dict_b = list(map(lambda x: x[1][0], pairs))
    post_a = list(map(lambda x: x[0][1], pairs))
    post_b = list(map(lambda x: x[1][1], pairs))
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(merge_itmd_index_postings, block_number, dict_a, post_a, dict_b, post_b):
            results.append(result)
    if len(block_names) % 2 == 1:
        results.append(block_names[len(block_names)-1])
    return merge_blocks(counter + len(results), num_docs, results)

def indexing(id_content_tuples):
    unigram_postings_dict = dict()
    for id_content_tuple in id_content_tuples:
        (docID, raw_content) = id_content_tuple
        content = re.sub(r'\s{2,}', ' ', raw_content)
        content = remove_html_css_js(content)
        processed_terms_list = preprocess_string(content)
        term_to_tf_dict = dict()
        for term, _ in processed_terms_list:
            if term not in processed_terms_list:
                term_to_tf_dict[term] = 0
            term_to_tf_dict[term] += 1
        mag_doc_vec = sqrt(reduce(lambda x, y: x + y**2, term_to_tf_dict.values(), 0))
        for term, position in processed_terms_list:
            normalized_w_td = log_tf(term_to_tf_dict[term]) / mag_doc_vec
            if term not in unigram_postings_dict:
                unigram_postings_dict[term] = list()
            unigram_posting_term_length = len(unigram_postings_dict[term])
            if unigram_posting_term_length != 0 and unigram_postings_dict[term][unigram_posting_term_length - 1][0] == docID:
                positions = unigram_postings_dict[term][unigram_posting_term_length - 1][1]
                positions.append(position)
                unigram_postings_dict[term][unigram_posting_term_length - 1] = (docID, positions, normalized_w_td)
            else:
                unigram_postings_dict[term].append((docID, [position], normalized_w_td))
    return unigram_postings_dict

def invert(block_number, document_chunk):
    unigram_postings_dict = indexing(document_chunk)
    block_index = Index()
    posting_file_name = "postings{}.txt".format(block_number)
    dictionary_file_name = "dictionary{}.txt".format(block_number)
    postings_file = open(posting_file_name, 'wb')
    dictionary_file = open(dictionary_file_name, 'wb')
    for term in unigram_postings_dict:
        offset = postings_file.tell()
        postings_byte = pickle.dumps(unigram_postings_dict[term])
        postings_size = sys.getsizeof(postings_byte)
        block_index.add_term_entry(term, offset, postings_size, len(unigram_postings_dict[term]))
        postings_file.write(postings_byte)
    del unigram_postings_dict
    postings_file.close()
    pickle.dump(block_index, dictionary_file)
    dictionary_file.close()
    return dictionary_file_name, posting_file_name

def merge_itmd_index_postings(block_number, dict_a_name, post_a_name, dict_b_name, post_b_name):
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
        new_dict.add_term_entry(term, offset, postings_size, len(new_post))
    new_post_fp.close()
    pickle.dump(new_dict, open(new_dict_name,"wb"))
    return (new_dict_name, new_post_name)

def set_idf(output_file_dictionary, num_docs):
    index = load_index(output_file_dictionary)
    for term in index:
        (offset, size, tf) = index.get_term_info(term)
        index.set_idf(term, offset, size, idf(tf,num_docs))
    pickle.dump(index, open(output_file_dictionary, "wb"))

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

def load_docID_to_court_dict():
    """
    Returns a dictionary that maps docIDs to courts.
    """
    return pickle.load(open('docID-court.txt', 'rb'))

def load_citation_to_docID_dict():
    """
    Returns a dictionary mapping neutral a citation to docID.
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
    string = re.sub(r'[^a-zA-Z\s]', '', string)
    tokens = word_tokenize(string)

    processed_tokens = []
    for token in tokens:
        if not is_stopword(token):
            processed_tokens.append(preprocess_string.lmtzr.lemmatize(preprocess_string.stemmer.stem(token)))
            # processed_tokens.append(preprocess_string.lmtzr.lemmatize(preprocess_string.stemmer.stem(spell(token))))

    return list(zip(processed_tokens, range(len(processed_tokens))))

def remove_html_css_js(raw_string):
    s = re.compile(r'^\/\/<!\[CDATA.*\]\]>',re.MULTILINE|re.DOTALL)
    return re.sub(s,'',raw_string)

def load_whole_dataset_csv(input_directory):
    # Reference: https://stackoverflow.com/a/15063941
    max_int = sys.maxsize
    should_decrement = True
    while should_decrement:
        try:
            csv.field_size_limit(max_int)
            should_decrement = False
        except OverflowError:
            max_int = int(max_int / 2)

    df = csv.reader(open(input_directory,"r"))
    next(df)
    return df

def get_citation(raw_string):
    """
    Returns the neutral citation of a law report's content (string).
    Returns `None` if no citation is found.
    """
    get_citation.re = r'\[\d+\] (\d+ )?[A-Z](\.*[A-Z]+)* \d+'
    match_obj = re.search(get_citation.re, raw_string[:200])

    if (match_obj == None):
        return None

    return match_obj.group(0)

def preprocess_docs(df):
    """
    Extracts:
    1. A list of (docID, content) tuples
    2. A mapping of {citation: docID}
    3. A mapping of {docID: court}
    """
    doc_id_set = set()
    docID_to_court_dict = dict()
    tuples = []
    for doc in df:
        doc_id = int(doc[0].lstrip('"').rstrip('"'))
        if doc[0] not in doc_id_set:
            doc_id_set.add(doc[0])
            tuples.append((doc_id, doc[1] + ' ' + doc[2]))
        docID_to_court_dict[doc_id] = Court.hierarchy[doc[4]]
    tuples.sort()

    with open('docID-court.txt', 'wb') as docID_to_court_file:
        pickle.dump(docID_to_court_dict, docID_to_court_file)

    citation_to_docID_dict = dict()
    for (docID, content) in tuples:
        citation = get_citation(content)
        if (citation != None):
            citation_to_docID_dict[citation] = docID
    with open('citation-docID.txt', 'wb') as citation_to_docID_file:
        pickle.dump(citation_to_docID_dict, citation_to_docID_file)

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
    df = load_whole_dataset_csv(input_directory)
    id_content_tuples = preprocess_docs(df)
    num_docs = len(id_content_tuples)
    logger.log_end_loading_dataset(num_docs)

    num_docs_per_block = 1000
    document_chunks = [id_content_tuples[i * num_docs_per_block:(i + 1) * num_docs_per_block] for i in range((num_docs + num_docs_per_block - 1) // num_docs_per_block )]
    # # Testing code to check invert code
    # invert(99, document_chunks[0])
    block_file_names = index_by_chunks(document_chunks)
    logger.log_start_merge_blocks()
    final_files = merge_blocks(len(block_file_names), num_docs, block_file_names)
    logger.log_end_merge_blocks()

    if (os.path.exists(output_file_dictionary)):
        os.remove(output_file_dictionary)
    if (os.path.exists(output_file_postings)):
        os.remove(output_file_postings)
    os.rename(final_files[0], output_file_dictionary)
    os.rename(final_files[1], output_file_postings)

    logger.log_start_calculating_idf()
    set_idf(output_file_dictionary, num_docs)
    logger.log_end_calculating_idf()

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

    def add_term_entry(self, term, offset, size, tf):
        self.__dict[term] = (offset, size, tf)

    def set_idf(self, term, offset, size, idf):
        self.__dict[term] = (offset, size, idf)

    def get_term_info(self, term):
        return self.__dict[term]

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