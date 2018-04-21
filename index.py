import concurrent.futures
import getopt
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


def invert(block_number, document_chunk):
    print("Get doc ID to unigram mapping")
    docID_to_unigrams_dict = get_docID_to_terms_mapping(document_chunk)
    print("Build unigram postings")
    unigram_postings_dict = build_unigram_postings(docID_to_unigrams_dict, list(map(lambda x: x[0], document_chunk)))
    block_index = Index()
    posting_file_name = "postings{}.txt".format(block_number)
    print("Writing...")
    postings_file = open(posting_file_name, 'wb')
    for term in unigram_postings_dict:
        offset = postings_file.tell()
        postings_byte = pickle.dumps(unigram_postings_dict[term])
        postings_size = sys.getsizeof(postings_byte)
        block_index.add_term_entry(term, offset, postings_size)
        postings_file.write(postings_byte)

    del unigram_postings_dict
    postings_file.close()
    return posting_file_name

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

    id_content_tuples = load_whole_dataset_csv(input_directory)
    num_docs = len(id_content_tuples)

    ## TODO: Bring back citation

    num_docs_per_block = 1000
    document_chunks = [id_content_tuples[i * num_docs_per_block:(i + 1) * num_docs_per_block] for i in range((num_docs + num_docs_per_block - 1) // num_docs_per_block )]
    document_block_postings = []
    print(invert(0, document_chunks[0]))
    # with concurrent.futures.ProcessPoolExecutor() as executor: # Put back the code first
    #     for result in executor.map(invert, list(range(len(document_chunks))), document_chunks):
    #         print("Finished indexing block: {}".format(result))
    #         document_block_postings.append(result)

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