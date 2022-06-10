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
                    help='tuples input file', required=True)
parser.add_argument('--fairness-qry-path', action='store', dest='fairness_qry_path',
                    help='path to fairness sensitive queries', required=True)
parser.add_argument('--out-file', action='store', dest='out_file',
                    help='tuples output file', required=True)


args = parser.parse_args()


qrys = {}
with open(args.fairness_qry_path, 'r') as fr:
    for line in fr:
        vals = line.strip().split('\t')
        qryid = vals[0]
        qrytext = vals[1]
        qrys[qryid] = qrytext
print (len(qrys))


with open(args.in_file, "r", encoding="utf8") as fr, open(args.out_file, "w", encoding="utf8") as fw:
    for line in tqdm(fr):
        vals = line.strip().split('\t')
        if len(vals) != 4:
            print("skipped %s" % line)
            continue
            
        query_id = vals[0].strip()
        if query_id in qrys:
            fw.write(line)
            
            