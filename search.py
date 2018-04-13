#!/usr/bin/python
import sys
import getopt
import itertools
from collections import OrderedDict
from index import load_dictionary, load_citation_to_docID_dict, get_idf, preprocess_string

from nltk.corpus import wordnet as wn

class Query:
    def __init__(self, query, expansion, positional=False):
        self._query = query
        self._expansion = expansion
        self._positional = positional
    
    def print_self(self):
        print(self._query, self._expansion, self._positional)

class QueryParser:
    def parse_query(self, query):
        queries = self._split_query_to_tokens(query)
        expanded_query = []
        for q in queries:
            syns = self._query_expansion(q.split(" "))
            syns = self._preprocess(syns)
            processed_query = self._preprocess(q)
            expanded_query.append(self._create_query_obj(query, processed_query, syns))
        return expanded_query
    
    def _is_boolean_query(self, query):
        """
        Check if the query is a boolean query
        """
        return " AND " in query

    def _split_query_to_tokens(self, query):
        """
        If the query is a boolean query, split it by the phrase.
        If the query is a free text query, just split it by spaces.
        """
        if self._is_boolean_query(query):
            return [q.strip().lstrip('"').rstrip('"') for q in query.split("AND")]
        else:
            return [q.strip() for q in query.split(" ")]

    def _query_expansion(self, list_of_queries):
        syns = []
        for query in list_of_queries:
            lemmas = list(map(lambda syn: syn.lemmas(), wn.synsets(query)[:2]))
            lemmas = list(itertools.chain(*lemmas))
            syns += [str(lemma.name()) for lemma in lemmas]
        syns = list(map(lambda syn: syn.replace("_", " "), syns))
        return syns

    def _preprocess(self, raw):
        """
        Interface to interact with preprocess_string, the return type is a string or a list,
        depending on what the input type is
        """
        things_to_process = raw
        if type(raw).__name__ == 'list':
            things_to_process = " ".join(raw)
        processed = list(OrderedDict.fromkeys(preprocess_string(things_to_process)))
        if type(raw).__name__ != 'list':
            processed = " ".join(processed)
        return processed

    def _create_query_obj(self, original_query, processed_query, syns):
        position_matters = self._is_boolean_query(original_query)
        return Query(processed_query, syns, position_matters)


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q query-file -o output-file-of-results")

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

    query_fp = open(file_of_queries, "r")
    query_parser = QueryParser()
    for line in query_fp:
        query_list = query_parser.parse_query(line)


##################################
# Procedural Program Starts Here #
##################################

if (__name__ == '__main__'):
    main()