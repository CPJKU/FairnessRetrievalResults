
## Data preperation

This base of this code is taken from (DeepGenIR)[https://github.com/CPJKU/DeepGenIR] repository. Please first follow the steps in that repository to prepare data files. 

Next, in order to create query-document tuples only for the fairness sensitive queries, execute the following commands:
```
cd collection_preparation
python tuples_filter_fairness_queries.py --in-file [PATH-TO-DEV-TUPLES] --fairness-qry-path ../../dataset/msmarco_passage.dev.fair.tsv --out-file [PATH-TO-DEV-TUPLES-NEW]
bash generate_file_split.sh [PATH-TO-DEV-TUPLES-NEW] 4 [PATH-TO-DEV-TUPLES-NEW].split-4/
```



## Usage
First, edit configs/msmarco-passge.yaml based on your file paths

Sample run commands for different usecases:

### Base model (BERT)

```python main.py --config-file configs/jku-msmarco-passage.yaml --cuda --gpu-id 0 --mode base --run-name base_l2```

### Training AdvBERT

```python main.py --config-file configs/msmarco-passage.yaml --cuda --gpu-id 0 --pretrained-model-folder [PATH] --config-overwrites "early_stopping_patience: -1, learning_rate_scheduler_patience: -1, adv_rev_factor: 1.0" --mode debias --run-name debias_tiny```

### Adversarial Attack

```python main.py --config-file configs/msmarco-passage.yaml --cuda --gpu-id 0 --pretrained-model-folder [PATH] --config-overwrites "early_stopping_patience: -1, learning_rate_scheduler_patience: -1, adv_rev_factor: 1.0" --mode attack --run-name attack_tiny```

 
 
