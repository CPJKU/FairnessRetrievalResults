import argparse
import os
import sys
from tqdm import tqdm
import pdb


#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='input file', required=True)
parser.add_argument('--out-file1', action='store', dest='out_file1',
                    help='output file', required=True)
parser.add_argument('--out-file2', action='store', dest='out_file2',
                    help='output file', required=True)
parser.add_argument('--out-file3', action='store', dest='out_file3',
                    help='output file', required=True)


args = parser.parse_args()


#
# load data (tokenize) & write out lines
# -------------------------------
#  

quota1 = 0
quota2 = 0
quota3 = 0

with open(args.in_file, "r", encoding="utf8") as fr, open(args.out_file1, "w", encoding="utf8") as fw1, open(args.out_file2, "w", encoding="utf8") as fw2, open(args.out_file3, "w", encoding="utf8") as fw3:
    for line in tqdm(fr):
        vals = line.strip().split('\t')
        if len(vals) != 6:
            print("skipped %s" % line)
            continue
        query_text = vals[0]
        doc_pos_text = vals[1]
        doc_neg_text = vals[2]
        query_gen_dist = vals[3]
        doc_pos_gen_dist = vals[4]
        doc_neg_gen_dist = vals[5]
        
        ## gender labels
        query_gen_dist = eval(query_gen_dist)
        doc_pos_gen_dist = eval(doc_pos_gen_dist)
        doc_neg_gen_dist = eval(doc_neg_gen_dist)
        
        if (query_gen_dist[0] + query_gen_dist[1]) > 0:
            continue
            
        if (doc_pos_gen_dist[0] + doc_pos_gen_dist[1] > 0):
            gender_label_pos = 1
        else:
            gender_label_pos = 0
        if (doc_neg_gen_dist[0] + doc_neg_gen_dist[1]) > 0:
            gender_label_neg = 1
        else:
            gender_label_neg = 0
        
        gendered_count = gender_label_pos + gender_label_neg
        nongendered_count = 2 - gendered_count
        
        # just count positives
        if (gender_label_pos == 1) or (gender_label_neg == 1):
            quota1 += gendered_count
            quota2 += gendered_count
            quota3 += gendered_count
            fw1.write(line)
            fw2.write(line)
            fw3.write(line)
        else:
            if quota1 > 0:
                fw1.write(line)
                quota1 -= nongendered_count
                continue
                
            if quota2 > 0:
                fw2.write(line)
                quota2 -= nongendered_count
                continue
                
            if quota3 > 0:
                fw3.write(line)
                quota3 -= nongendered_count
                continue



            