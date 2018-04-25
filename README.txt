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
We started off with using N-gram index. However, we later found that the number of terms (i.e. unigrams, bigrams, trigrams) becomes too large, making the entire index files too large. To save space, positional index is used, where by the structure of postings are changed to (docID, [pos1, pos2, ...], tf-idf). Changing from N-gram index to positional index reduced the size of `dictionary.txt` from 1.2 Gb to 7.5 Mb and the size of `postings.txt` from more than 2 Gb to roughly 1 Gb.

In addition to the above, we also decided to make our our system's phrasal search more accommodating by appending the results of the phrasal search with the results obtained by treating the phrasal queries as free text queries.
For example, to answer a phrasal query of "fertility treatment", the top results would first be documents with "fertility treatment" as a phrase, followed by documents with "fertility" and/or "treatment".

** Ranked Phrasal Query **
Phrasal queries are ranked by first doing one round of boolean retrieval, and subsequently computing the cosine similarity of all discovered document to the query and subsequently sort them in non-increasing score.

** Law Report Citation **
Law report citations are unique for each case and lawyers often search for law reports using citations. As citations play such a central role in law report searching, the law report citations are extracted to facilitate searches.

** Court Hierarchy **
The corpus of this homework assignment consists of law reports from courts of the United Kingdom, Australia, Singapore, Hong Kong, and the US. Different courts have different levels of authority depending on the position of the court in the country's court hierarchy. A court with higher authority therefore has more influential reports, which can be factored into the way search results are ranked.

We searched for the court hierarchies for all the countries mentioned aboved, and for each country, we assigned 'authority values' to the courts from between 0 to 1, where 1 is usually the highest court. These values are moderated across countries such that the authority values of courts of different countries are comparable.
During search, the cosine similarity values for a document is multiplied by the 'authority value' of its corresponding court before sorting the search result.

** Autocorrect (Experimental) **
During indexing, it was discovered that the documents in dataset.csv has many spelling errors, such as 'ludqment', 'distlnction', 'llkellhood', etc. We attempted to use the `autocorrect` package to do the corrections. However, the experiment that attempts to correct each and every words in the corpus took more than 24 hours to index. Therefore, correcting spelling is impractical.



== Files included with this submission ==

index.py        - Performs indexing
search.py       - Performs searching
dictionary.txt  - Pickled dictionary mapping terms to (offset, size, idf)
postings.txt    - Pickled postings
citation-docID.txt  - Pickled dictionary mapping law report citations to document ID
docID-court.txt - Pickled dictionary mapping document ID to courts
BONUS.docx      - Outlines and discusses the query expansion techniques used in this homework assignment

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
