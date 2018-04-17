#!/usr/bin/python
import sys
import getopt
import itertools
from math import sqrt
from functools import reduce
from collections import OrderedDict, Counter
from index import load_index, load_citation_to_docID_dict, log_tf, preprocess_string, get_postings
from index import Index

from nltk.corpus import wordnet as wn

class Query:
    def __init__(self, query, expansion, positional=False):
        self._query = query
        self._expansion = expansion
        self._positional = positional
    
    def position_matters(self):
        return self._positional

    def get_query(self):
        return self._query
    
    def get_expansion(self):
        return self._expansion
    
    def print_self(self):
        print(self._query, self._expansion, self._positional)

class QueryParser:
    def parse_query(self, query):
        queries = self._split_query_to_tokens(query)
        expanded_query = []
        all_syns = []
        all_processed_query = []
        for q in queries:
            syns = self._query_expansion(q.split(" "))
            syns = self._preprocess(syns)
            all_syns.extend(syns)
            processed_query = self._preprocess(q)
            all_processed_query.append(processed_query)
            if self._is_boolean_query(query):
                expanded_query.append(self._create_query_obj(query, processed_query, syns))
        if not self._is_boolean_query(query):
            expanded_query.append(self._create_query_obj(query, " ".join(all_processed_query), all_syns))
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

class SearchEngine:
    def __init__(self, index, postings):
        self._index = index
        self._postings = postings

    def search(self, query_list):
        result = []
        for query_obj in query_list:
            result.append(self._search_query(query_obj))
        if len(result) > 1:
            result.sort(key=lambda t: len(t))
            result = self._boolean_retrieval_and(result)
        else:
            result = result[0]
        return result
    
    def _search_query(self, query):
        if query.position_matters():
            return self._boolean_retrieval(query)
        else:
            return self._free_text_query(query)

    def _boolean_retrieval(self, query):
        return list(map(lambda x : str(x[0]), get_postings(query.get_query(), self._index, self._postings)))

    def _boolean_retrieval_and(self, result):
        # TODO: Skip pointers
        if len(result) == 1 or len(result[0]) == 0:
            return result[0]
        else:
            and_result = []
            a_point = 0
            b_point = 0
            while a_point < len(result[0]) and b_point < len(result[1]):
                a = result[0][a_point]
                b = result[1][b_point]
                if int(a) == int(b):
                    and_result.append(a)
                    a_point += 1
                    b_point += 1
                elif int(a) < int(b):
                    a_point += 1
                else:
                    b_point += 1
            result.pop(0)
            result.pop(0)
            result.insert(0, and_result)
            return self._boolean_retrieval_and(result)

    def _free_text_query(self, query):
        query_tokens = query.get_query().split(" ") + query.get_expansion()
        query_vector = self._compute_query_vector(query_tokens)
        document_vectors = self._compute_document_vectors(query_tokens)
        return self._free_text_score(document_vectors, query_vector)

    def _compute_query_vector(self, query_tokens):
        term_to_tf_dict = dict(Counter(query_tokens))
        term_to_w_td_dict = {}
        for (term, tf) in term_to_tf_dict.items():
            w_td = log_tf(tf) * self._index.get_idf(term)
            term_to_w_td_dict[term] = w_td
        normalize = sqrt(reduce(lambda x, y: x + y**2, term_to_w_td_dict.values(), 0))
        normalized_term_to_w_td_dict = dict(map(lambda term_to_w_td: (term_to_w_td[0], term_to_w_td[1]/normalize), term_to_w_td_dict.items()))
        return normalized_term_to_w_td_dict

    def _compute_document_vectors(self, query_tokens):
        doc_dict = {}
        for token in query_tokens:
            if token in self._index:
                postings = get_postings(token, self._index, self._postings)
                for post in postings:
                    doc_id = post[0]
                    w_td = post[1]
                    if doc_id not in doc_dict:
                        doc_dict[doc_id] = {}
                    doc_dict[doc_id][token] = w_td
        return doc_dict
    
    def _free_text_score(self, document_vectors, query_vector):
        docs_score = []
        for (doc_id, document_vector) in document_vectors.items():
            doc_score = 0
            for token in document_vector:
                doc_score += document_vector[token] * query_vector[token]
            docs_score.append((doc_id, doc_score))
        docs_score.sort(key=lambda x : (-x[1], x[0]))
        return list(map(lambda x: str(x[0]), docs_score))

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
    index = load_index(dictionary_file)
    posting_file = open(postings_file, "rb")
    output_fp = open(file_of_output, "w")
    search_engine = SearchEngine(index, posting_file)
    query_parser = QueryParser()
    for line in query_fp:
        query_list = query_parser.parse_query(line)
        # for q in query_list:
        #     q.print_self()
        output_fp.writelines(" ".join(search_engine.search(query_list)))
        # for q in query_list:
        #     q.print_self()
    output_fp.close()


##################################
# Procedural Program Starts Here #
##################################

if (__name__ == '__main__'):
    main()