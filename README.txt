This is the README file for A0126623L-A0126576X's submission

== Python Version ==

We're using Python Version 3.6.4 for
this assignment.

== General Notes about this assignment ==

** Scalable Indexing **
One challenge of this homework is the indexing of the large corpus consisting more than 17,000 documents. A modified SPIMI was implemented, where by there is the notion of 'blocks' or 'chunks'. Each chunks consists of 1000 documents. We index the documents chunk-by-chunk, storing the block dictionary and block postings as intermediate files, and finally merging them in the end.

** Query Expansion **
Query expansion and pseudo relevance feedback were used to expand the query to obtain more results. This is discussed in greater detail in BONUS.docx.

** Phrasal Query **
Two methods were considered and implemented for answering phrasal queries.
    i) N-gram index
    ii) Positional index
We started off with using N-gram index. However, we later found that the number of terms (i.e. unigrams, bigrams, trigrams) becomes too large, making the entire index files too large. To save space, positional index is used, where by the structure of postings are changed to (docID, [pos1, pos2, ...], tf-idf).

** Ranked Phrasal Query **
Phrasal queries are ranked by first doing one round of boolean retrieval, and subsequently computing the cosine similarity of all discovered document to the query and subsequently sort them in non-increasing score.

== Files included with this submission ==

index.py        - Performs indexing
search.py       - Performs searching
dictionary.txt  - Pickled dictionary mapping terms to (offset, size, idf)
postings.txt    - Pickled postings
citation-docID.txt  - Pickled dictionary mapping law report citations to document ID
docID-court.txt - Pickled dictionary mapping document ID to courts
BONUS.docx      - Outlines the query expansion techniques used in this homework assignment

== Statement of individual work ==

Please initial one of the following statements.

[x] I, A0126623L-A0126576X, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

I suggest that I should be graded as follows:

<Please fill in>

== References ==

Textbook:
"Introduction to Information Retrieval", written by Christopher D Manning, Prabhakar Raghavan, Hinrich Schutze
