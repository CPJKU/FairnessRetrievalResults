import argparse
import os
import sys
from tqdm import tqdm
import pdb

from document_neutrality import DocumentNeutrality

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='input file', required=True)
parser.add_argument('--threshold', action='store', type=int, default=1,
                    help='Threshold for the number of representative words in the text')
parser.add_argument('--representative-words-path', action='store', dest='representative_words_path',
                    default="../../resources/wordlists/wordlist_gender_representative.txt",
                    help='path to the list of representative words which define the protected attribute')
parser.add_argument('--fair-queries-path', action='store', dest='fair_queries_path',
                    default="../../resources/fairnesssensitive_queries/msmarco_passage.dev.fair.tsv",
                    help='path of the fair queries')
parser.add_argument('--out-file', action='store', dest='out_file',
                    help='output file', required=True)

args = parser.parse_args()



## reading fair queries
fair_queries = []
with open(args.fair_queries_path, "r", encoding="utf8") as fr:
    for line in fr:
        fair_queries.append(int(line.split('\t')[0]))
fair_queries = set(fair_queries)
print (fair_queries)

## calculating document neutrality scores
_doc_neutrality = DocumentNeutrality(args.representative_words_path, threshold=args.threshold)
_query_neutrality = DocumentNeutrality(args.representative_words_path, threshold=0)


with open(args.out_file, "w", encoding="utf8") as fw:
    with open(args.in_file, "r", encoding="utf8") as fr:
        for line in tqdm(fr):
            vals = line.strip().lower().split('\t')
            if len(vals) != 4:
                print("skipped %s" % line)
                continue
            query_id = vals[0]
            doc_id = vals[1]
            query_text = vals[2]
            doc_text = vals[3]
            
            if int(query_id) in fair_queries:
                neut_qry = _query_neutrality.get_neutrality(query_text)
                neut_doc = _doc_neutrality.get_neutrality(doc_text)

                fw.write("%s\t%s\t%s\t%s\t%f\t%f\n" % (query_id, doc_id, query_text, doc_text, neut_qry, neut_doc))



            