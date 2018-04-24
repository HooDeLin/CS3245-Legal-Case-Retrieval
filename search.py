#!/usr/bin/python
import sys
import getopt
import itertools
from math import sqrt
from functools import reduce
from collections import OrderedDict, Counter
from index import load_index, log_tf, preprocess_string, get_postings, load_citation_to_docID_dict, load_docID_to_court_dict
from index import Index

from nltk.corpus import wordnet as wn

PSEUDO_RELEVANCE_FEEDBACK_TOP_K = 10
PSEUDO_RELEVANCE_RATIO = 0.3
COURT_ORDER_RELEVANCE_RATIO = 0.1 # We don't want the court order to be too influential

# Toggling this three constants will set different configurations for searching
QUERY_EXPANSION = True
PSEUDO_RELEVANCE_FEEDBACK = True
COURT = True

class Query:
    """
    Query class is an abstraction for a query, which records the expansion version of the query,
    as well as if the query is a boolean retrieval query
    """
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
    """
    The Query Parser accepts the raw query and create Query objects
    It is able to determine whether the query is a boolean retrieval query
    or a free text query
    """
    def parse_query(self, query):
        """
        The function expands the query, and creates the query object
        """
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
        """
        The function expands the query
        """
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
        processed = list(map(lambda x: x[0], OrderedDict.fromkeys(preprocess_string(things_to_process))))
        if type(raw).__name__ != 'list':
            processed = " ".join(processed)
        return processed

    def _create_query_obj(self, original_query, processed_query, syns):
        position_matters = self._is_boolean_query(original_query)
        return Query(processed_query, syns, position_matters)

class SearchEngine:
    """
    The SearchEngine class is the main search mechanism to interpret the index and postings
    as well as getting the result from the query objects
    """
    def __init__(self, index, postings, docID_to_court_dict):
        self._index = index
        self._postings = postings
        self._docID_to_court_dict = docID_to_court_dict

    def search(self, query_list):
        result = []
        query_tokens = [] # Query Tokens are needed for boolean retrieval
        for query_obj in query_list:
            query_tokens.extend(query_obj.get_query().split(" "))
            result.append(self._search_query(query_obj))
        if len(result) > 1: # This only happens during boolean retrieval
            result.sort(key=lambda t: len(t))
            result = self._boolean_retrieval_and(result)
            # Rank boolean retrieval result:
            # We get the result doc ids and construct the document vectors
            # And use the document vectors to get the cosine similarity
            # To rank the results
            query_vector = self._compute_query_vector(query_tokens)
            document_vectors = self._get_document_vectors_from_id(result, query_tokens)
            return self._free_text_score(document_vectors, query_vector)
        else:
            result = result[0]
        return result
    
    def _search_query(self, query):
        if query.position_matters():
            return self._boolean_retrieval(query)
        else:
            return self._free_text_query(query)

    def _boolean_retrieval(self, query):
        """
        For every query, we get the query string postings.
        After that we do a positional merging.
        """
        query_string_postings = list(map(lambda x: get_postings(x, self._index, self._postings), query.get_query().split(" ")))
        if sum(list(map(lambda x: len(x), query_string_postings))) == 0:
            return []
        result = self._positional_merging(query_string_postings)
        return list(map(lambda x : str(x[0]), result))
    
    def _positional_merging(self, result):
        if len(result) == 1 or len(result[0]) == 0: # We return the result if there is only one result left
            return result[0]
        else:
            and_result = []
            a_point = 0
            b_point = 0
            while a_point < len(result[0]) and b_point < len(result[1]):
                a = result[0][a_point]
                b = result[1][b_point]
                if a[0] == b[0]: # compare document ids
                    new_positions = self._intersect_positions(a[1],b[1])
                    # We need to check if the positions are consecutive
                    # Don't continue if positions are not executive
                    if len(new_positions) != 0:
                        and_result.append((b[0], new_positions, b[2]))
                    a_point += 1
                    b_point += 1
                elif int(a[0]) < int(b[0]):
                    a_point += 1
                else:
                    b_point += 1
            result.pop(0)
            result.pop(0)
            result.insert(0, and_result)
            return self._positional_merging(result) # We do this recursively
    
    def _intersect_positions(self, a_positions, b_positions):
        """
        We return a list of b_positions where a_positions are just before b_positions
        """
        new_positions = []
        for a_position in a_positions:
            if a_position + 1 in b_positions:
                new_positions.append(a_position + 1)
        return new_positions

    def _boolean_retrieval_and(self, result):
        """
        This is a AND query among query results
        """
        if len(result) == 1 or len(result[0]) == 0: # We return the result if there is only one result left
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
        """
        This is the main function for free text query
        We expand the query if needed, and then construct the query vector
        From the query tokens, we get the document vector that has the terms
        After that, we can update the query vector using pseudo-relevance feedback
        Finally we use the cosine similarity between the document_vectors and query_vector
        """
        query_tokens = query.get_query().split(" ")
        if QUERY_EXPANSION:
            query_tokens += query.get_expansion()
        query_vector = self._compute_query_vector(query_tokens)
        document_vectors = self._compute_document_vectors(query_tokens)
        if PSEUDO_RELEVANCE_FEEDBACK:
            query_vector = self._create_prf_query_vector(query_vector, document_vectors)
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
        """
        We construct a dictionary of document_vectors that has any query_tokens
        """
        doc_dict = {}
        for token in query_tokens:
            if token in self._index:
                postings = get_postings(token, self._index, self._postings)
                for post in postings:
                    doc_id = post[0]
                    w_td = post[2]
                    if doc_id not in doc_dict:
                        doc_dict[doc_id] = {}
                    doc_dict[doc_id][token] = w_td
        return doc_dict
    
    def _get_document_vectors_from_id(self, ids, query_tokens):
        """
        From a list of document ids, we would want to get the document vectors for each of them
        """
        id_list = set(map(lambda x: int(x), ids))
        doc_dict = {}
        for token in query_tokens:
            if token in self._index:
                postings = get_postings(token, self._index, self._postings)
                for post in postings:
                    doc_id = post[0]
                    w_td = post[2]
                    if doc_id in id_list:
                        if doc_id not in doc_dict: # Only get document vector if it's in the list
                            doc_dict[doc_id] = {}
                        doc_dict[doc_id][token] = w_td
        return doc_dict

    def _create_prf_query_vector(self, query_vector, document_vectors):
        """
        From a list of document vectors, we calculated how much we need to shift the original
        query vector. This will make sure that the query vector is shifted towards the centroid
        """
        relevant_doc_ids = list(map(lambda x: x[0], self._generate_doc_score_tuple(query_vector, document_vectors, top_n=PSEUDO_RELEVANCE_FEEDBACK_TOP_K, court_relevance=False)))
        # Calculate the adjustments for each term in the vector
        adjustments = {}
        for doc_id in relevant_doc_ids:
            for vector_component in document_vectors[doc_id]:
                if vector_component not in adjustments:
                    adjustments[vector_component] = 0
                adjustments[vector_component] += document_vectors[doc_id][vector_component]
        prf_query_vector = {}
        for vector_component in query_vector:
            adj = 0
            if vector_component in adjustments:
                # We need to do this because query tokens that has 0 score will appear 
                # at query_vector but no in adjustments
                adj = adjustments[vector_component]
            new_value = (1-PSEUDO_RELEVANCE_RATIO) * query_vector[vector_component] + (PSEUDO_RELEVANCE_RATIO * adj) / PSEUDO_RELEVANCE_FEEDBACK_TOP_K
            prf_query_vector[vector_component] = new_value
        return prf_query_vector

    
    def _free_text_score(self, document_vectors, query_vector):
        docs_score = self._generate_doc_score_tuple(query_vector, document_vectors, court_relevance=True)
        return list(map(lambda x: str(x[0]), docs_score))
    
    def _generate_doc_score_tuple(self, query_vector, document_vectors, top_n=None, court_relevance=False):
        """
        For each document vector, we calculate the cosine similarity of it with query vector
        """
        docs_score = []
        for (doc_id, document_vector) in document_vectors.items():
            doc_score = 0
            for token in document_vector:
                doc_score += document_vector[token] * query_vector[token]
            if court_relevance and COURT:
                docs_score.append((doc_id, (1-COURT_ORDER_RELEVANCE_RATIO) * doc_score + COURT_ORDER_RELEVANCE_RATIO * self._docID_to_court_dict[doc_id]))
            else:
                docs_score.append((doc_id, doc_score))
        docs_score.sort(key=lambda x : (-x[1], x[0]))
        if top_n == None:
            return docs_score
        else:
            return docs_score[:top_n]

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q query-file -o output-file-of-results")

def parse_input_arguments():
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
    return (dictionary_file, postings_file, file_of_queries, file_of_output)

def main():
    (dictionary_file, postings_file, file_of_queries, file_of_output) = parse_input_arguments()
    query_fp = open(file_of_queries, "r")
    index = load_index(dictionary_file)
    posting_file = open(postings_file, "rb")
    output_fp = open(file_of_output, "w")
    docID_to_court_dict = load_docID_to_court_dict()
    citation_to_doc_dict = load_citation_to_docID_dict()
    search_engine = SearchEngine(index, posting_file, docID_to_court_dict)
    query_parser = QueryParser()
    for line in query_fp:
        query = line.strip()
        result = ""
        # We skip boolean retrieval and free text query if the query is a citation
        if query in citation_to_doc_dict:
            result = str(citation_to_doc_dict[query])
        else:
            query_list = query_parser.parse_query(query)
            result = " ".join(search_engine.search(query_list))
        output_fp.writelines(result)
    output_fp.close()


##################################
# Procedural Program Starts Here #
##################################

if (__name__ == '__main__'):
    main()