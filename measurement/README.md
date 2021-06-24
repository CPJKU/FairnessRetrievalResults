# Fairness of Retrieval Result (FaiRR - NFaiRR) Metric

The files in this folder provide everything necessary for calculating FaiRR and NFaiRR metrics on any given retrieval results. The code can be used to measure FaiRR and NFaiRR on any TREC-formatted run file. The provided classes can also be simply used inside an existing Python code to calculate the fairness metrics for some retrieval results.

## Step 1: Calculate & Store Document Neutrality Scores 
To this end, first, you should use `calculate_collection_docneutrality.py` to calculate and store the gender neutrality scores of the documents in the collection. This file reads each document in the collection and calculates its gender neutrality score. The scores are then stored in a file (default: `processed/collection_neutralityscores.tsv`), which are then used to calculate the fairness metrics. Here a sample code to run the script:
```
python3 calc_documents_neutrality.py --collection-path [PATH_TO_TSV_COLLECTION] --representative-words-path ../resources/wordlists/wordlist_gender_representative.txt --threshold 1 --out-file processed/collection_neutralityscores.tsv
```

Please consider that the current code expects the collection to be in one TSV file, as for instance provided in MS MARCO collection. Also, the code applies no pre-processing (only `.lower()`) and tokenizes the documents with simple white space spliting. Covering other formats/cases requires adaptation in code. However, the only important output of this step is the stored output file.   

## Step 2: Calculate Fairness Metrics

Given any TREC-formatted run file, the fairness metrics can be calculated with the following command:
```
python metrics_fairness.py --collection-neutrality-path processed/collection_neutralityscores.tsv --backgroundrunfile sample_trec_runs/msmarco_passage/BM25.run --runfile sample_trec_runs/msmarco_passage/advbert_L4.run --print-qry-results
```
