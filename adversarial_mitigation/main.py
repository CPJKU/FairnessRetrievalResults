#
# train a neural-ir model
# -------------------------------
#
# features:
#
# * uses pytorch + allenNLP
# * tries to correctly encapsulate the data source (msmarco)
# * clear configuration with yaml files
#
# usage:
# python train.py --run-name experiment1 --config-file configs/knrm.yaml

import argparse
import copy
import os
import pdb
import pickle
import glob
import time
import sys
import numpy as np
sys.path.append(os.getcwd())
from typing import Dict, Tuple, List

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import *
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from allennlp.common import Params, Tqdm
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.nn.util import move_to_device

from transformers import BertModel, BertConfig, AutoConfig, BartForConditionalGeneration, BartConfig
from transformers import BertTokenizer

from model import *
from optimizers import Optimizer
from utils import *
from evaluation import *
from multiprocess_input_pipeline import *
from fairness_measurement.metrics_fairness import FaiRRMetric, FaiRRMetricHelper
from metrics_utility import EvaluationToolTrec, EvaluationToolMsmarco

Tqdm.default_mininterval = 1

EPS = 1e-20
MASK_SYMBOL = "[MASK]"


def evaluate_validation():
    global best_result_info
    
    _output_relative_dir = os.path.join(run_folder, "checkpoints/chk%d-%d/" % (epoch, batch_cnt_global))
    if not os.path.exists(_output_relative_dir):
        os.makedirs(_output_relative_dir)
    
    _result_info, _qry_doc_relscores = evaluate_model(model, config, logger, run_folder, cuda_device,
                                                      evaluator=evaluator_val,
                                                      evaluator_fairness=evaluator_fairness,
                                                      reference_set_rank=reference_set_rank_val, 
                                                      reference_set_tuple=reference_set_tuple_val,
                                                      output_files_prefix="",
                                                      output_relative_dir=_output_relative_dir,
                                                      testval="validation")
    
    for _m in _result_info["metrics_avg"]:
        tb_writer.add_scalar("val/%s" % _m, _result_info["metrics_avg"][_m], batch_cnt_global)

    _metric = config["metric_tocompare"]
    if ((args.mode in ['debias', 'attack']) or (best_result_info is None) or 
        (_result_info["metrics_avg"][_metric] > best_result_info["metrics_avg"][_metric])):
        
        best_result_info = _result_info
        
        # save validation results
        _best_result_output_path = os.path.join(run_folder, "validation-best-run.txt")
        _best_result_info_path = os.path.join(run_folder, "validation-best-metrics.txt")
        _best_result_info_pkl_path = os.path.join(run_folder, "validation-best-metrics.pkl")
        
        save_sorted_results(_qry_doc_relscores, _best_result_output_path)    

        with open(_best_result_info_path, "w") as fw:
            fw.write("{'metrics_avg':%s, 'epoch':%d, 'batch_number':%d}" % 
                     (str(best_result_info["metrics_avg"]), epoch, batch_cnt_global))
        
        with open(_best_result_info_pkl_path, "wb") as fw:
            pickle.dump(best_result_info, fw)
        
        # save (best) validation model state
        logger.info("Saving new model with %s of %.4f at %s" % (_metric, 
                                                                best_result_info['metrics_avg'][_metric],
                                                                best_model_store_path))
        model_save(best_model_store_path, model, best_result_info)

    return _result_info
        
#
# main process
# -------------------------------
#
if __name__ == "__main__":
    
    ###############################################################################
    # Inititialization
    ###############################################################################

    #
    # config
    # -------------------------------
    #
    args = get_parser().parse_args()

    run_folder, config = prepare_experiment(args)
    
    tb_writer = SummaryWriter(run_folder)

    logger = get_logger_to_file(run_folder, "main")

    logger.info("Running: %s", str(sys.argv))
    logger.info("Experiment folder: %s", run_folder)
    logger.info("Config vars: %s", str(config))

    tb_writer.add_text("info", "Config vars: %s" % str(config))

    logger.info('-' * 89)
    logger.info("Starting experiment %s", args.run_name)
    logger.info('-' * 89)

    # Set the random seed manually for reproducibility.
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning("You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed_all(config["seed"])


    if args.cuda:
        # hardcode gpu usage
        cuda_device = int(args.cuda_device_id)
        torch.cuda.init() # just to make sure we can select a device
        torch.cuda.set_device(cuda_device)
        logger.info("Using cuda device id: %i - %s", cuda_device, torch.cuda.get_device_name(cuda_device))
    else:
        cuda_device = -1


    ###############################################################################
    # Evaluation 
    ###############################################################################

    # utility
    
    logger.info('Loading validation qrels and reference set')
    reference_set_rank_val, reference_set_tuple_val = parse_reference_set(config["validation_candidate_set_path"],
                                                                          config["evaluation_reranking_cutoff"])
    evaluator_val = EvaluationToolMsmarco(qrel_path=config["validation_qrels"])
    #evaluator_val = EvaluationToolTrec(trec_eval_path=config["trec_eval_path"], qrel_path=config["validation_qrels"])
    
    logger.info('Loading test qrels and reference set')
    reference_set_rank_test, reference_set_tuple_test = parse_reference_set(config["test_candidate_set_path"],
                                                                            config["evaluation_reranking_cutoff"])
    evaluator_test = EvaluationToolTrec(trec_eval_path=config["trec_eval_path"], qrel_path=config["test_qrels"])

    # fairness
    
    _metrichelper = FaiRRMetricHelper()
    _background_doc_set = _metrichelper.read_documentset_from_retrievalresults(config["background_runfile_path"])
    evaluator_fairness = FaiRRMetric(config["collection_neutrality_path"], _background_doc_set)
    
    ###############################################################################
    # Load data 
    ###############################################################################

    logger.info("Loading BERT")
    if config["transformers_pretrained_model_id"] == "random":
        _bert_config_standard = AutoConfig.for_model('bert')
        bert_config = BertConfig(vocab_size = _bert_config_standard.vocab_size,
                                 hidden_size = config["bert_hidden_size"], #?
                                 num_hidden_layers = config["bert_num_layers"],
                                 num_attention_heads = config["bert_num_heads"],
                                 intermediate_size = config["bert_intermediate_size"],
                                 hidden_dropout_prob = config["bert_dropout"], #!
                                 attention_probs_dropout_prob = config["bert_dropout"])
        transformer_embedder = BertModel(bert_config)
    else:
        transformer_embedder = BertModel.from_pretrained(config["transformers_pretrained_model_id"], cache_dir="cache")
    _bert_tokenizer = BertTokenizer.from_pretrained(config["transformers_pretrained_model_id"])
    _bert_cls_token_id = _bert_tokenizer.vocab[_bert_tokenizer.cls_token]
    _bert_sep_token_id = _bert_tokenizer.vocab[_bert_tokenizer.sep_token]

    ###############################################################################
    # Model
    ###############################################################################

    model = AdvBert(bert = transformer_embedder, adv_rev_factor=config["adv_rev_factor"],
                    cls_token_id=_bert_cls_token_id, sep_token_id=_bert_sep_token_id)
    
    if cuda_device != -1:
        model.cuda(cuda_device)

    logger.info('Model: %s', model)
    logger.info('Model total parameters: %s', sum(p.numel() for p in model.parameters()))
    logger.info('Model total trainable parameters: %s', sum(p.numel() for p in model.parameters() if p.requires_grad))

    ###############################################################################
    # Train and Validation
    ###############################################################################
    best_model_store_path = os.path.join(run_folder, "model.best.pt")
    checkpoint_model_store_path = os.path.join(run_folder, "model.checkpoint.pt")


    if args.mode in ['debias', 'attack', 'base']:
        
        #
        # loading parameters from other pre-trained models
        #
        if args.pretrained_model_folder is not None:
            _pretrained_model_path = os.path.join(args.pretrained_model_folder, 'model.best.pt')
            logger.info('Loading pretrained parameters from %s' % _pretrained_model_path)
            
            pretrained_model_state, _ = model_load(_pretrained_model_path, cuda_device)
            model_state = model.state_dict()
            _loaded_params_name = []
            for _param_name, _param in pretrained_model_state.items():
                if _param_name in model_state:
                    # in the 'attack' mode, let fresh adversary parameter be created
                    if (args.mode == 'attack') and ("adversary_net" in _param_name):
                        continue
                    
                    if isinstance(_param, torch.nn.Parameter):
                        # backwards compatibility for serialized parameters
                        _param = _param.data
                    model_state[_param_name].copy_(_param)
                    _loaded_params_name.append(_param_name)
            logger.info('Loaded pretrained parameter state_dict %s' % str(_loaded_params_name))

        #
        # optimization
        #    
        if config["optimizer"] == "adam":
            params_group_model = []
            params_group_adv = []

            _added_params_name_model = []
            _added_params_name_adv = []
            for p_name, par in model.named_parameters():
                if "adversary" not in p_name:
                    params_group_model.append(par)
                    _added_params_name_model.append(p_name)
                else:
                    params_group_adv.append(par)
                    _added_params_name_adv.append(p_name)
            logger.info('Parameters added to MODEL optimizer %s' % str(_added_params_name_model))
            logger.info('Parameters added to ADVERSARY optimizer %s' % str(_added_params_name_adv))

            optimizer_model = optim.Adam([{"params":params_group_model, "lr":config["param_group_model_learning_rate"],
                                           "weight_decay":config["param_group_model_weight_decay"]}], 
                                         betas=(0.9, 0.999), eps=0.00001)
            optimizer_adv = optim.Adam([{"params":params_group_adv, "lr":config["param_group_adversary_learning_rate"],
                                           "weight_decay":config["param_group_adversary_weight_decay"]}], 
                                         betas=(0.9, 0.999), eps=0.00001)

        lr_scheduler = None
        if config["learning_rate_scheduler_patience"] != -1:
            lr_scheduler = ReduceLROnPlateau(optimizer_model, mode="max",
                                             patience=config["learning_rate_scheduler_patience"],
                                             factor=config["learning_rate_scheduler_factor"],
                                             verbose=True)
        early_stopper = None
        if config["early_stopping_patience"] != -1:
            early_stopper = EarlyStopping(patience=config["early_stopping_patience"], mode="max", min_delta=0.0005)   


        #
        # loss
        #   
        criterion = None
        if config["loss_model"] == "maxmargin":
            criterion = torch.nn.MarginRankingLoss(margin=config["loss_model_maxmargin_margin"], reduction='mean')
        elif config["loss_model"] == "crossentropy":
            pass

        criterion_adversary = None
        criterion_adversary = torch.nn.NLLLoss()
        
        if (cuda_device != -1):
            if (criterion is not None):
                criterion.cuda(cuda_device)
            if (criterion_adversary is not None):
                criterion_adversary.cuda(cuda_device)
        
        #
        # training / saving / validation loop
        # -------------------------------
        #
        
        logger.info('-' * 89)
        logger.info('Training started...')
        logger.info('-' * 89)

        best_result_info = None 
        training_batch_size = int(config["batch_size_train"])
        validate_every_n_batches = config["validate_every_n_batches"]
        # helper vars for quick checking if we should validate during the epoch
        do_validate_every_n_batches = validate_every_n_batches > -1
        loss_sum_model = 0
        loss_sum_adv = 0
        data_cnt_all = 0
        training_processes = []
        batch_cnt_global = 0

        try:
            for epoch in range(0, int(config["epochs"])):
                if early_stopper is not None:
                    if early_stopper.stop:
                        break
                #
                # data loading
                # -------------------------------
                #
                train_files = glob.glob(config.get("train_tsv"))
                training_queue, training_processes, train_exit = get_multiprocess_batch_queue("train-batches-" + str(epoch),
                                                                                              multiprocess_training_loader,
                                                                                              files=train_files,
                                                                                              conf=config,
                                                                                              _logger=logger)
                
                model.train()  # only has an effect, if we use dropout & regularization layers in the model definition...
                #time.sleep(len(training_processes))  # fill the queue
                logger.info("[Epoch %d] --- Start training with queue.size:%d" % (epoch, training_queue.qsize()))
                
                #
                # train loop
                # -------------------------------
                #
                i = 0
                batch_null_cnt = 0
                max_training_batch_count = config["max_training_batch_count"]
                
                # do validation at the begining
                if "save_test_during_validation" in config:
                    if config["save_test_during_validation"]:
                        evaluate_validation()    

                while (True):
                    
                    #
                    # prepare batch
                    #
                    batch = training_queue.get()
                    if batch is None:
                        batch_null_cnt += 1
                        if batch_null_cnt == len(train_files):
                            break
                        else:
                            continue
                    if i >= max_training_batch_count and max_training_batch_count != -1:
                        break
                    batch_cnt_global += 1

                    if isinstance(batch["query_tokens"], dict):
                        current_batch_size = int(batch["query_tokens"]["tokens"].shape[0])
                    else:
                        current_batch_size = int(batch["query_tokens"].shape[0])

                    data_cnt_all += current_batch_size

                    if cuda_device != -1:
                        batch = move_to_device(batch, cuda_device)

                    #
                    # feed forward
                    #
                    optimizer_model.zero_grad()
                    optimizer_adv.zero_grad()

                    output_pos_dict = model.forward(batch["query_tokens"], batch["doc_pos_tokens"])
                    output_pos = output_pos_dict["rels"]

                    output_neg_dict = model.forward(batch["query_tokens"], batch["doc_neg_tokens"])
                    output_neg = output_neg_dict["rels"]
                    
                    #
                    # loss & optimization
                    #
                    loss_model = None 
                    if args.mode in ['debias', 'base']:
                        if config["loss_model"] == "maxmargin":
                            labels = torch.ones(current_batch_size)
                            if cuda_device != -1:
                                labels = labels.cuda(cuda_device)
                            loss_model = criterion(output_pos, output_neg, labels)
                        elif config["loss_model"] == "crossentropy":
                            outputs = output_pos_dict["logprobs"][:,0] + output_neg_dict["logprobs"][:,1]
                            loss_model = - torch.mean(outputs)
                        else:
                            logger.error("Model loss function %s not known", config["loss_model"])
                            exit(1)


                    loss_adv = None    
                    if args.mode in ['debias', 'attack']:
                        loss_adv = criterion_adversary(output_pos_dict["adv_logprobs"], batch["protected_label_pos"])
                        loss_adv += criterion_adversary(output_neg_dict["adv_logprobs"], batch["protected_label_neg"])

                    if loss_model is None:
                        loss = loss_adv
                    elif loss_adv is None:
                        loss = loss_model
                    else:
                        loss = loss_model + loss_adv

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    if args.mode in ['debias', 'base']:
                        optimizer_model.step()
                    if args.mode in ['debias', 'attack']:
                        optimizer_adv.step()

                        
                    #
                    # logging
                    #
                    if loss_model is not None:
                        loss_sum_model += loss_model.item()
                    if loss_adv is not None:
                        loss_sum_adv += loss_adv.item()
                    
                    tb_writer.add_scalar("train/loss", loss.item(), batch_cnt_global)
                    if (i % config["log_interval"] == 0) and (i != 0):
                        cur_loss_model = loss_sum_model / float(data_cnt_all)
                        cur_loss_adv = loss_sum_adv / float(data_cnt_all)
                        
                        logger.info('| TRAIN | %s | epoch %3d | %5d batches | lrs %s %s | avg. loss %.5f %.5f' %
                                    (args.run_name, epoch, i, str(['%02.6f' % pg['lr'] for pg in optimizer_model.param_groups]), 
                                     str(['%02.6f' % pg['lr'] for pg in optimizer_adv.param_groups]), 
                                     cur_loss_model, cur_loss_adv))

                        # make sure that the perf of the queue is sustained
                        if training_queue.qsize() < 10:
                            logger.warning("training_queue.qsize() < 10 (%d)" % training_queue.qsize())
                        
                    if config["checkpoint_interval"] != -1 and i % config["checkpoint_interval"] == 0 and i > 0:
                        logger.info("saving checkpoint at epoch %3d and %5d batches" % (epoch, i))
                        checkpoint_save(checkpoint_model_store_path, model, criterion, optimizer_model, epoch, i)

                    #
                    # validation (inside epoch) - if so configured
                    #
                    if do_validate_every_n_batches:
                        if i > 0 and i % validate_every_n_batches == 0:
                            _result_info = evaluate_validation()
                            
                            # coming back to training mode
                            model.train()
                            if lr_scheduler is not None:
                                lr_scheduler.step(_result_info["metrics_avg"][config["metric_tocompare"]])
                            if early_stopper is not None:
                                if early_stopper.step(_result_info["metrics_avg"][config["metric_tocompare"]]):
                                    logger.info("early stopping epoch %d batch count %d" % (epoch, i))
                                    break
                            
                    i += 1 #next batch

                # make sure we didn't make a mistake in the configuration / data preparation
                if training_queue.qsize() != 0 and config["max_training_batch_count"] == -1:
                    logger.error("training_queue.qsize() is not empty after epoch "+str(epoch))

                ## logging loss
                cur_loss_model = loss_sum_model / float(data_cnt_all)
                cur_loss_adv = loss_sum_adv / float(data_cnt_all)
                logger.info('| TRAIN | %s | epoch %3d FINISHED | %5d batches | lrs %s %s | avg. loss %.5f %.5f' %
                            (args.run_name, epoch, i, str(['%02.6f' % pg['lr'] for pg in optimizer_model.param_groups]), 
                             str(['%02.6f' % pg['lr'] for pg in optimizer_adv.param_groups]),
                             cur_loss_model, cur_loss_adv))
                
                if config["checkpoint_interval"] != -1:
                    logger.info("saving checkpoint at epoch %d after %d batches" % (epoch, i))
                    checkpoint_save(checkpoint_model_store_path, model, criterion, optimizer_model, epoch, i)

                #terminating sub processes
                train_exit.set()  # allow sub-processes to exit
                for proc in training_processes:
                    if proc.is_alive():
                        proc.terminate()

                #
                # validation (at the end of epoch)
                #
                _result_info = evaluate_validation()
                if lr_scheduler is not None:
                    lr_scheduler.step(_result_info["metrics_avg"][config["metric_tocompare"]])
                if early_stopper is not None:
                    if early_stopper.step(_result_info["metrics_avg"][config["metric_tocompare"]]):
                        logger.info("early stopping epoch %d" % epoch)
                        break

        except Exception as e:
            logger.exception('[train] Got exception: %s' % str(e))
            logger.info('Exiting from training early')

            for proc in training_processes:
                if proc.is_alive():
                    proc.terminate()
            exit(1)

        logger.info('Training Finished!')
        logger.info('=' * 89)
      
    ###############################################################################
    # Test
    ###############################################################################
    if args.mode in ['debias', 'attack', 'base', 'test']:
        #
        # evaluate the test set with the best model
        #
        logger.info('-' * 89)
        logger.info('Test set evaluation started...')
        logger.info("Experiment: %s", args.run_name)
        logger.info('-' * 89)

        logger.info('Loading best model')

        _cuda_device = int(args.cuda_device_id)
        model_state, best_result_info_val = model_load(best_model_store_path, _cuda_device)
        model.load_state_dict(model_state)

        logger.info("Testing the model")
        
        _result_info, _ = evaluate_model(model, config, logger, run_folder, cuda_device,
                                         evaluator=evaluator_test,
                                         evaluator_fairness=evaluator_fairness,
                                         reference_set_rank=reference_set_rank_test, 
                                         reference_set_tuple=reference_set_tuple_test,
                                         output_files_prefix=config["test_files_prefix"],
                                         output_relative_dir="",
                                         testval="test")
        
        for _m in _result_info["metrics_avg"]:
            tb_writer.add_scalar("test/%s" % _m, _result_info["metrics_avg"][_m], 0)


    
    logger.info('Fertig!')
    
