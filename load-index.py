#!/usr/bin/python
import sys
import getopt
import pickle

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q query-file -o output-file-of-results")

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
def load_dictionary(dictionary_file):
    return pickle.load(open(dictionary_file, 'rb'))

def get_postings(term, dictionary, postings_reader):
    """
    Parameters
        dictionary: A dictionary mapping 'terms' -> (offset, idf)
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

def main():
    dictionary_file = postings_file = file_of_queries = output_file_of_results = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except getopt.GetoptError as err:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-d':
            dictionary_file  = a
        elif o == '-p':
            postings_file = a
        elif o == '-q':
            file_of_queries = a
        elif o == '-o':
            file_of_output = a
        else:
            assert False, "unhandled option"

    if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
        usage()
        sys.exit(2)

    dictionary = load_dictionary(dictionary_file)
    postings_reader = open(postings_file, 'rb')

    terms = ['road', 'origin', 'option']
    for term in terms:
        postings = get_postings(term, dictionary, postings_reader)
        print("'{}' --> {}\n".format(term, postings))

##################################
# Procedural Program Starts Here #
##################################
if (__name__ == '__main__'):
    main()

