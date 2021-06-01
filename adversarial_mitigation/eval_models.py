import os
import copy
import time
import glob
from typing import Dict, Tuple, List
import pdb
import numpy as np
import pickle
import torch
from pathlib import Path

from allennlp.nn.util import move_to_device
from allennlp.common import Params, Tqdm

from sklearn.metrics import accuracy_score

from multiprocess_input_pipeline import *
from utils import *

METRICS = {'map', 'ndcg_cut', 'recip_rank', 'P', 'recall'}

def load_reference_from_stream(f):
    """Load Reference reference relevant documents
    """
    qids_to_relevant_docids = {}
    for l in f:
        vals = l.strip().split('\t')
        if len(vals) != 4:
            vals = l.strip().split(' ')
            if len(vals) != 4:
                raise IOError('\"%s\" is not valid format' % l)

        #qid = int(vals[0])
        qid = vals[0]
        docid = vals[2]
        score = int(vals[3])
        if qid not in qids_to_relevant_docids:
            qids_to_relevant_docids[qid] = {}
        if docid in qids_to_relevant_docids[qid]:
            raise Error("One doc can not have multiple relevance for a query. QID=%d, docid=%s" % (qid, docid))
        qids_to_relevant_docids[qid][docid] = score

    return qids_to_relevant_docids

def load_reference(path_to_reference):
    """Load Reference reference relevant documents
    """
    with open(path_to_reference,'r') as f:
        qids_to_relevant_docids = load_reference_from_stream(f)
    return qids_to_relevant_docids

def load_candidate_from_stream(f):
    """Load candidate data from a stream.
    """
    qid_to_ranked_candidate_docs = {}
    for l in f:
            l = l.strip().split()
            try:
                if len(l) == 4: # own format
                    #qid = int(l[0])
                    qid = l[0]
                    docid = l[1]
                    rank = int(l[2])
                    score = float(l[3])
                if len(l) == 6: # original trec format
                    #qid = int(l[0])
                    qid = l[0]
                    docid = l[2]
                    rank = int(l[3])
                    score = float(l[4])
            except:
                raise IOError('\"%s\" is not valid format' % l)
            if qid not in qid_to_ranked_candidate_docs:
                qid_to_ranked_candidate_docs[qid] = {}
            if docid in qid_to_ranked_candidate_docs[qid]:
                raise Exception("Cannot rank a doc multiple times for a query. QID=%s, docid=%s" % (str(qid), str(docid)))
            qid_to_ranked_candidate_docs[qid][docid] = score
    return qid_to_ranked_candidate_docs

def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    """

    with open(path_to_candidate,'r') as f:
        qid_to_ranked_candidate_docs = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_docs

def compute_metrics(evaluator, qids_to_ranked_candidate, candidate_path_for_save, save_perquery=True):
    
    # code for pytrec_eval library
    #results_perq = evaluator.evaluate(qids_to_ranked_candidate_docs)
    #_metrics = list(list(results_perq.values())[0].keys())
    #for _metric in _metrics:
    #    results_avg[_metric] = pytrec_eval.compute_aggregated_measure(_metric, [x[_metric] for x in results_perq.values()])
    
    #code for adhoc trec_eval
    _evalparam = "-q" if save_perquery else ""
    results_avg, results_perq = evaluator.evaluate(candidate=qids_to_ranked_candidate, 
                                                   run_path_for_save=candidate_path_for_save,
                                                   evalparam=_evalparam)
    
    return results_avg, results_perq

def compute_metrics_from_file(evaluator, path_to_candidate):
    qids_to_ranked_candidate_docs = load_candidate(path_to_candidate)
    
    results_avg, results_perq = evaluator.evaluate_from_file(candidate_path=path_to_candidate)
    
    result_info = {}
    result_info["metrics_avg"] = results_avg
    result_info["metrics_perq"] = results_perq
    result_info["qry_doc_relscores"] = qids_to_ranked_candidate_docs
    result_info["cs@n"] = -1

    return result_info

def compute_metrics_at_cutoff(evaluator, path_to_candidate, reference_set_rank, reference_set_tuple,
                              reference_set_cutoff):

    qids_to_ranked_candidate_docs = load_candidate(path_to_candidate)
    
    pruned_qids_to_ranked_candidate_docs = {}
    for qid in qids_to_ranked_candidate_docs:
        rank_docids = list(qids_to_ranked_candidate_docs[qid].items())
        rank_docids.sort(key=lambda x: x[1], reverse=True)
        rank_docids = [x[0] for x in rank_docids]

        pruned_qids_to_ranked_candidate_docs[qid] = {}
        added = 0
        added_docids = []
        for rank, docid in enumerate(rank_docids):
            if docid in reference_set_rank[qid] and reference_set_rank[qid][docid] <= reference_set_cutoff:
                pruned_qids_to_ranked_candidate_docs[qid][docid] = qids_to_ranked_candidate_docs[qid][docid]
                added_docids.append(docid)
                added += 1

        _scores = list(pruned_qids_to_ranked_candidate_docs[qid].values())
        if len(_scores) > 0:
            score_diff = np.min(_scores) - np.max([x[1] for x in reference_set_tuple[qid]])
        else:
            score_diff = np.max([x[1] for x in reference_set_tuple[qid]])
        # adding what is rest till i to the list if it is missing
        for _tuple in reference_set_tuple[qid]:
            docid, score = _tuple
            if reference_set_cutoff <= added:
                break
            if docid not in added_docids:
                pruned_qids_to_ranked_candidate_docs[qid][docid] = score + score_diff # make scores lower than "pruned"
                added += 1

        # adding the rest from i or added till the end
        for j in range(added, len(reference_set_tuple[qid])):
            docid, score = reference_set_tuple[qid][j]
            pruned_qids_to_ranked_candidate_docs[qid][docid] = score + score_diff

    candidate_path_for_save_pruned = path_to_candidate+'.pruned'
    #path_to_candidate_pruned = path_to_candidate+'.pruned'
    #save_sorted_results(pruned_qids_to_ranked_candidate_docs, path_to_candidate_pruned)
    metric_results_avg, metric_results_perq = compute_metrics(evaluator, pruned_qids_to_ranked_candidate_docs,
                                                              candidate_path_for_save_pruned)
        
    result_info = {}
    result_info["metrics_avg"] = metric_results_avg
    result_info["metrics_perq"] = metric_results_perq
    result_info["qry_doc_relscores"] = pruned_qids_to_ranked_candidate_docs
    result_info["cs@n"] = reference_set_cutoff

    return result_info


def save_sorted_results(results, file_path, until_rank=-1):
    with open(file_path, "w") as val_file:
        for qid in results.keys():
            query_data = list(results[qid].items())
            query_data.sort(key=lambda x: x[1], reverse=True)
            # sort the results per query based on the output
            for rank_i, (docid, score) in enumerate(query_data):
                #val_file.write("\t".join(str(x) for x in [query_id, doc_id, rank_i + 1, output_value])+"\n")
                val_file.write("%s Q0 %s %d %f neural\n" % (str(qid), str(docid), rank_i + 1, score))

                if until_rank > -1 and rank_i == until_rank + 1:
                    break

def save_adv_predictions(adv_predictions_labels, file_path):
    with open(file_path, "w") as fw:
        for qid in adv_predictions_labels:
            for docid in adv_predictions_labels[qid]:
                # qid docid prediction_label
                vals = adv_predictions_labels[qid][docid]
                fw.write("%s %s %d\n" % (str(qid), str(docid), vals[0]))


#
# raw model evaluation, returns model results as python dict, does not save anything / no metrics
#
def predict_relevance(model, cuda_device, eval_tsv, config, logger):

    model.eval()  # turning off training
    qry_doc_relscores = {}
    protected_predictions_labels = {}
    _processes = []
    
    _max_batch_count = config["max_evaluation_batch_count"]
    _log_interval = config["eval_log_interval"]
        
    try:
        _files = glob.glob(eval_tsv)
        _queue, _processes, _exit = get_multiprocess_batch_queue("eval-batches",
                                                                 multiprocess_validation_loader,
                                                                 files=_files,
                                                                 conf=config,
                                                                 _logger=logger,
                                                                 queue_size=200)
        batch_num = 0
        batch_null_cnt = 0

        with torch.no_grad():
            while (True):
                batch_orig = _queue.get()
                if batch_orig is None:
                    batch_null_cnt += 1
                    if batch_null_cnt == len(_files):
                        break
                    else:
                        continue
                if batch_num >= _max_batch_count and _max_batch_count != -1:
                    break

                batch = copy.deepcopy(batch_orig)
                if cuda_device != -1:
                    batch = move_to_device(batch, cuda_device)
                
                output_dict = model.forward(batch["query_tokens"], batch["doc_tokens"])
                
                rels = output_dict["rels"]
                rels = rels.cpu()  # get the relevance scores back to the cpu - in one piece
                if "adv_logprobs" in output_dict:
                    adv_logprobs = output_dict["adv_logprobs"]
                    adv_logprobs = adv_logprobs.cpu()  # get the relevance scores back to the cpu - in one piece

                for sample_i, sample_query_id in enumerate(batch_orig["query_id"]):  # operate on cpu memory
                    sample_query_id = sample_query_id
                    sample_doc_id = batch_orig["doc_id"][sample_i]  # again operate on cpu memory
                    sample_protected_label = batch_orig["protected_label"][sample_i]  # again operate on cpu memory

                    if sample_query_id not in qry_doc_relscores:
                        qry_doc_relscores[sample_query_id] = {}
                        protected_predictions_labels[sample_query_id] = {}

                    qry_doc_relscores[sample_query_id][sample_doc_id] = float(rels[sample_i])
                    
                    if "adv_logprobs" in output_dict:
                        _predicted_protected_label = int(torch.argmax(adv_logprobs[sample_i]))
                    else:
                        _predicted_protected_label = 0
                    protected_predictions_labels[sample_query_id][sample_doc_id] = (_predicted_protected_label,
                                                                                    sample_protected_label)
                    
                
                if batch_num % _log_interval == 0:
                    logger.info('INFERENCE | %5d batches' % (batch_num))
                    if _queue.qsize() < 10:
                        logger.warning("evaluation_queue.qsize() < 10 (%d)" % _queue.qsize())

                batch_num += 1

        logger.info('INFERENCE FINISHED | %5d batches ' % (batch_num))
        
        # make sure we didn't make a mistake in the configuration / data preparation
        if _queue.qsize() != 0 and _max_batch_count == -1:
            logger.error("evaluation_queue.qsize() is not empty (%d) after evaluation" % _queue.qsize())

        _exit.set()  # allow sub-processes to exit

        for proc in _processes:
            if proc.is_alive():
                proc.terminate()

    except BaseException as e:
        logger.exception('[eval_model] Got exception: %s' % str(e))

        for proc in _processes:
            if proc.is_alive():
                proc.terminate()
        raise e

    return qry_doc_relscores, protected_predictions_labels

#
# evaluate a model + save results and metrics 
#
def evaluate_model(model, config, logger, run_folder, cuda_device, evaluator,
                   reference_set_rank, reference_set_tuple, output_files_prefix, output_relative_dir="", testval="val"):

    logger.info("[INFERENCE] --- Start")

    qry_doc_relscores, protected_predictions_labels = predict_relevance(model, cuda_device, config["%s_tsv" % testval], 
                                                                        config, logger)

    #
    # save full rerank results
    #
    logger.info("Saving file with prefix: " + output_files_prefix)
    fullrerank_path = os.path.join(run_folder, output_relative_dir, "%s%s-run-full-rerank.txt" % (output_files_prefix, testval))
    save_sorted_results(qry_doc_relscores, fullrerank_path)

    #
    # compute evaluation metrics 
    # ---------------------------------
    #
    _cutoff = config["evaluation_reranking_cutoff"]
    result_info = compute_metrics_at_cutoff(evaluator, fullrerank_path, reference_set_rank, reference_set_tuple,
                                            reference_set_cutoff=config["evaluation_reranking_cutoff"])
    
    # save evaluated rank list
    _path = os.path.join(run_folder, output_relative_dir, "%s%s-run.txt" % (output_files_prefix, testval)) 
    save_sorted_results(result_info["qry_doc_relscores"], _path)

    result_info_tosave = {}
    for _key in result_info:
        result_info_tosave[_key] = result_info[_key]

    #
    # save & evaluate the results of the fairness metric
    #
    
    # accuracy of prediction in the adversary head
    _adv_predictions = []
    _adv_labels = []
    for qid in protected_predictions_labels:
        for docid in protected_predictions_labels[qid]:
            _adv_predictions.append(protected_predictions_labels[qid][docid][0])
            _adv_labels.append(protected_predictions_labels[qid][docid][1])
            
    _accuracy = accuracy_score(_adv_predictions, _adv_labels)
    _accuracy_dummybaseline = 1 - (np.sum(_adv_labels) / float(len(_adv_labels)))
    
    result_info_tosave["metrics_avg"]["adv_accuracy"] = _accuracy
    result_info_tosave["metrics_avg"]["adv_accuracy_dummybaseline"] = _accuracy_dummybaseline
    
    logger.info("ADVERSARY accuracy: %f" % (_accuracy))
    logger.info("ADVERSARY dummy baseline accuracy: %f" % (_accuracy_dummybaseline))

    _path = os.path.join(run_folder, output_files_prefix + "adversarial-predictions.txt")
    save_adv_predictions(protected_predictions_labels, _path)
    
    # fairness metrics
    # TODO
    
    # save final results
    logger.info("Results: %s" % (result_info_tosave["metrics_avg"]))
    
    _path = os.path.join(run_folder, output_relative_dir, "%s%s-metrics.txt" % (output_files_prefix, testval)) 
    with open(_path, "w") as fw:
        fw.write(str(result_info_tosave["metrics_avg"]))
    _path = os.path.join(run_folder, output_relative_dir, "%s%s-metrics.pkl" % (output_files_prefix, testval)) 
    with open(_path, "wb") as fw:
        pickle.dump(result_info_tosave, fw)

    return result_info
