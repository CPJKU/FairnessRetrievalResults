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
parser.add_argument('--out-file', action='store', dest='out_file',
                    help='output file', required=True)

args = parser.parse_args()


#
# load data (tokenize) & write out lines
# -------------------------------
#  

_doc_neutrality = DocumentNeutrality(args.representative_words_path, threshold=args.threshold)
_query_neutrality = DocumentNeutrality(args.representative_words_path, threshold=0)


with open(args.out_file, "w", encoding="utf8") as fw:
    with open(args.in_file, "r", encoding="utf8") as fr:
        for line in tqdm(fr):
            vals = line.strip().lower().split('\t')
            if len(vals) != 3:
                print("skipped %s" % line)
                continue
            query_text = vals[0]
            doc_pos_text = vals[1]
            doc_neg_text = vals[2]
            
            neut_qry = _query_neutrality.get_neutrality(query_text)
            neut_docpos = _doc_neutrality.get_neutrality(doc_pos_text)
            neut_docneg = _doc_neutrality.get_neutrality(doc_neg_text)
            
            fw.write("%s\t%s\t%s\t%f\t%f\t%f\n" % (query_text, doc_pos_text, doc_neg_text, neut_qry, neut_docpos, neut_docneg))



        



            