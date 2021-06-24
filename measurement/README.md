# Fairness of Retrieval Results (FaiRR) Metric



First, calculate the neutrality scores for the documents in collection:
```
python3 calc_documents_neutrality.py --collection-path [PATH_TO_TSV_COLLECTION] --representative-words-path ../resources/wordlists/wordlist_gender_representative.txt --threshold 1 --out-file processed/collection_neutralityscores.tsv
```

Calculating fairness metrics for a specific TREC run file:
```
python metrics_fairness.py --collection-neutrality-path processed/collection_neutralityscores.tsv --backgroundrunfile sample_trec_runs/msmarco_passage/BM25.run --runfile sample_trec_runs/msmarco_passage/advbert_L4.run --print-qry-results
```
