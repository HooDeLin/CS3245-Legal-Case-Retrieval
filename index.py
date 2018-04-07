#!/usr/bin/python
import re
import os
import sys
import getopt
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

    df = pd.read_csv(input_directory, index_col=0)
    
    

##################################
# Procedural Program Starts Here #
##################################
if (__name__ == '__main__'):
    main()

    print("index.py finished running! :)")  # TODO: Remove before submission
    
