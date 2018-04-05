#!/usr/bin/python
import sys
import getopt
import itertools
from nltk.corpus import wordnet as wn

def parse_and_expand_query(query):
    queries = [q.strip().lstrip('"').rstrip('"') for q in query.split("AND")]
    expanded_query = []
    for query in queries:
        query_tokens = query.split(" ")
        syns = []
        for q in query_tokens:
            lemmas = list(map(lambda syn: syn.lemmas(), wn.synsets(q)[:2]))
            lemmas = list(itertools.chain(*lemmas))
            syns += [str(lemma.name()) for lemma in lemmas]
        syns = list(map(lambda syn: syn.replace("_", " "), syns))
        expanded_query.append((query, list(set(syns))))
    return expanded_query

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q query-file -o output-file-of-results")

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

for line in query_fp:
    print(parse_and_expand_query(line))