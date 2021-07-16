import argparse
import numpy as np
import pickle
import pdb
import itertools
import copy

class FaiRRMetric:
    
    def __init__(self, collection_neutrality_path, background_doc_set, thresholds=[5,10,20,50]):
        self.documents_neutrality = {}
        for l in open(collection_neutrality_path):
            vals = l.strip().split('\t')
            self.documents_neutrality[int(vals[0])] = float(vals[1])
        self.background_doc_set = background_doc_set
        self.thresholds = thresholds
        
        self.position_biases = [1/(np.log2(_rank+1)) for _rank in range(1, 1001)]


        ## get neutrality of background documents
        _bachgroundset_neut = {}
        for _qryid in self.background_doc_set:
            _bachgroundset_neut[_qryid] = []
            for _docid in self.background_doc_set[_qryid]:
                if _docid in self.documents_neutrality:
                    _neutscore = self.documents_neutrality[_docid]
                else:
                    _neutscore = 1.0
                    print("WARNING: Document neutrality score of ID %d is not found (set to 1)" % _doc_id)
                _bachgroundset_neut[_qryid].append(_neutscore)
        
        ## calculate Ideal FaiRR
        self.IFaiRR_perq = {}
        for _qryid in _bachgroundset_neut:
            _bachgroundset_neut[_qryid].sort(reverse=True)
        for _threshold in self.thresholds:
            self.IFaiRR_perq[_threshold] = {}
            for _qryid in _bachgroundset_neut:
                _th = np.min([len(_bachgroundset_neut[_qryid]), _threshold])
                self.IFaiRR_perq[_threshold][_qryid] = np.sum(np.multiply(_bachgroundset_neut[_qryid][:_th], 
                                                                          self.position_biases[:_th]))
            
        
    # the normalization term IFaiRR is calculated using the documents of the to retrieval_results
    # retrieval_results : a dictionary with queries and the ordered lists of documents
    def calc_FaiRR_retrievalresults(self, retrievalresults):
        
        ## get neutrality of documents
        _retres_neut = {}
        for _qryid in retrievalresults:
            _retres_neut[_qryid] = []
            for _docid in retrievalresults[_qryid][:np.max(self.thresholds)]:
                if _docid in self.documents_neutrality:
                    _neutscore = self.documents_neutrality[_docid]
                else:
                    _neutscore = 1.0
                    print("WARNING: Document neutrality score of ID %d is not found (set to 1)" % _doc_id)
                _retres_neut[_qryid].append(_neutscore)
        
        ## calculate FaiRR
        FaiRR = {}
        FaiRR_perq = {}
        for _threshold in self.thresholds:
            FaiRR_perq[_threshold] = {}
            for _qryid in _retres_neut:
                _th = np.min([len(_retres_neut[_qryid]), _threshold])
                FaiRR_perq[_threshold][_qryid] = np.sum(np.multiply(_retres_neut[_qryid][:_th], self.position_biases[:_th]))
            FaiRR[_threshold] = np.mean(list(FaiRR_perq[_threshold].values()))

        ## calculate Normalized FaiRR
        NFaiRR = {}
        NFaiRR_perq = {}
        for _threshold in self.thresholds:
            NFaiRR_perq[_threshold] = {}
            for _qryid in FaiRR_perq[_threshold]:
                if _qryid not in self.IFaiRR_perq[_threshold]:
                    print("ERROR: query id %d does not exist in background document set. Error ignored" % _qryid)
                    continue
                NFaiRR_perq[_threshold][_qryid] = FaiRR_perq[_threshold][_qryid] / self.IFaiRR_perq[_threshold][_qryid]
            NFaiRR[_threshold] = np.mean(list(NFaiRR_perq[_threshold].values()))
        
        return {'metrics_avg': {'FaiRR': FaiRR, 'NFaiRR': NFaiRR}, 
                'metrics_perq': {'FaiRR': FaiRR_perq, 'NFaiRR': NFaiRR_perq}}
    
    
    # doc_set : a dictionary with queries and the set of documents
    def calc_FaiRR_rankeragnostic(self, doc_set_withqry):
        
        
        ## get neutrality of documents
        _docs_neut = {}
        for _qryid in doc_set_withqry:
            _docs_neut[_qryid] = []
            for _docid in doc_set_withqry[_qryid]:
                if _docid in self.documents_neutrality:
                    _neutscore = self.documents_neutrality[_docid]
                else:
                    _neutscore = 1.0
                    print("WARNING: Document neutrality score of ID %d is not found (set to 1)" % _doc_id)
                _docs_neut[_qryid].append(_neutscore)
        
        ## calculate FaiRR
        FaiRR = {}
        FaiRR_perq = {}
        for _th in self.thresholds:
            FaiRR[_th] = {}
            FaiRR_perq[_th] = {}
            for _qryid in _docs_neut:
                FaiRR_perq[_th][_qryid] = np.mean(_docs_neut[_qryid]) * np.sum(self.position_biases[:_th])
            FaiRR[_th] = np.mean(list(FaiRR_perq[_th].values()))
        
        ## calculate Normalized FaiRR
        NFaiRR = {}
        NFaiRR_perq = {}
        for _threshold in self.thresholds:
            NFaiRR_perq[_threshold] = {}
            for _qryid in FaiRR_perq[_threshold]:
                if _qryid not in self.IFaiRR_perq[_threshold]:
                    print("ERROR: query id %d does not exist in background document set. Error ignored" % _qryid)
                    continue
                NFaiRR_perq[_threshold][_qryid] = FaiRR_perq[_threshold][_qryid] / self.IFaiRR_perq[_threshold][_qryid]
            NFaiRR[_threshold] = np.mean(list(NFaiRR_perq[_threshold].values()))
        
        return {'metrics_avg': {'FaiRR': FaiRR, 'NFaiRR': NFaiRR}, 
                'metrics_perq': {'FaiRR': FaiRR_perq, 'NFaiRR': NFaiRR_perq}}
    
    # doc_set : a dictionary with queries and the set of documents
    def calc_FaiRR_rankeragnostic_collection(self, doc_set):
        
        ## get neutrality of documents
        _docs_neut = []
        for _docid in doc_set:
            if _docid in self.documents_neutrality:
                _neutscore = self.documents_neutrality[_docid]
            else:
                _neutscore = 1.0
                print("WARNING: Document neutrality score of ID %d is not found (set to 1)" % _doc_id)
            _docs_neut.append(_neutscore)
        
        ## calculate FaiRR
        FaiRR = {}
        for _th in self.thresholds:
            FaiRR[_th] = np.mean(_docs_neut) * np.sum(self.position_biases[:_th])
        
        ## calculate Normalized FaiRR
        NFaiRR = {}
        NFaiRR_perq = {}
        for _threshold in self.thresholds:
            NFaiRR_perq[_threshold] = {}
            for _qryid in self.IFaiRR_perq[_threshold]:
                NFaiRR_perq[_threshold][_qryid] = FaiRR[_threshold] / self.IFaiRR_perq[_threshold][_qryid]
            NFaiRR[_threshold] = np.mean(list(NFaiRR_perq[_threshold].values()))
        
        return {'metrics_avg': {'FaiRR': FaiRR, 'NFaiRR': NFaiRR}}

class FaiRRMetricHelper:

    def read_retrievalresults_from_runfile(self, trec_run_path, cut_off=200):
        retrievalresults = {}
        
        print ("Reading %s" % trec_run_path)

        with open(trec_run_path) as fr:
            qryid_cur = 0
            for i, line in enumerate(fr):
                vals = line.strip().split(' ')
                if len(vals) != 6:
                    vals = line.strip().split('\t')
                
                if len(vals) == 6:
                    _qryid = int(vals[0].strip())
                    _docid = int(vals[2].strip())
                    
                    if _qryid != qryid_cur:
                        retrievalresults[_qryid] = []
                        qryid_cur = _qryid

                    if len(retrievalresults[_qryid]) < cut_off:
                        retrievalresults[_qryid].append(_docid)
                else:
                    pass

        print ('%d lines read. Number of queries: %d' % (len(list(itertools.chain.from_iterable(retrievalresults.values()))), 
                                                         len(retrievalresults.keys())))
        
        return retrievalresults
    
    def read_documentset_from_retrievalresults(self, trec_run_path):
        _retrivalresults_background = self.read_retrievalresults_from_runfile(trec_run_path)
        background_doc_set = {}
        for _qryid in _retrivalresults_background:
            background_doc_set[_qryid] = set(_retrivalresults_background[_qryid])
        return background_doc_set


if __name__ == "__main__":
    #
    # config
    #
    parser = argparse.ArgumentParser()

    parser.add_argument('--collection-neutrality-path', action='store', dest='collection_neutrality_path',
                        default="processed/collection_neutralityscores.tsv",
                        help='path to the file containing neutrality values of documents in tsv format (docid [tab] score)')
    parser.add_argument('--backgroundrunfile', action='store',
                        default="sample_trec_runs/msmarco_passage/BM25.run",
                        help='path to the run file for the set of background documents in TREC format', required=True)
    parser.add_argument('--runfile', action='store', dest='runfile',
                        default="sample_trec_runs/msmarco_passage/advbert_L4.run",
                        help='path to the run file in TREC format. It can be ignored if --ignore-runfile is used',
                        required=False)
    parser.add_argument('--print-qry-results', action='store_true', dest='print_qry_results',
                        help='Print the results per query in addition to the average results')
    parser.add_argument('--ignore-runfile', action='store_true', dest='ignore_runfile',
                        help='Ignores run file and only calculates the ranker-agnostic metrics')
    args = parser.parse_args()
    
    _metric_helper = FaiRRMetricHelper()
    _background_doc_set = _metric_helper.read_documentset_from_retrievalresults(args.backgroundrunfile)
    print ("Reading document neutrality scores ...")
    _fairr_metric = FaiRRMetric(args.collection_neutrality_path, _background_doc_set)
    print ("Reading document neutrality scores ... done!")
    _retrivalresults = _metric_helper.read_retrievalresults_from_runfile(args.runfile)
    print ()
    
    print ("*** Ranker-agnostic fairness metrics for all documents in collection ***")
    _all_doc_set = set(_fairr_metric.documents_neutrality.keys())
    _metric_res = _fairr_metric.calc_FaiRR_rankeragnostic_collection(_all_doc_set)
    _ms = list(_metric_res['metrics_avg'].keys())
    _ms.sort()
    
    for _m in _ms:
        _cutoffs = list(_metric_res['metrics_avg'][_m].keys())
        _cutoffs.sort()
        for _cutoff in _cutoffs:
            print ("%s_%d All:" % (_m, _cutoff), _metric_res['metrics_avg'][_m][_cutoff])
    print ()
            
    print ("*** Ranker-agnostic fairness metrics for the documents taken from %s ***" % args.backgroundrunfile)
    _metric_res = _fairr_metric.calc_FaiRR_rankeragnostic(_background_doc_set)
    _ms = list(_metric_res['metrics_avg'].keys())
    _ms.sort()
    
    for _m in _ms:
        _cutoffs = list(_metric_res['metrics_avg'][_m].keys())
        _cutoffs.sort()
        for _cutoff in _cutoffs:
            print ("%s_%d All:" % (_m, _cutoff), _metric_res['metrics_avg'][_m][_cutoff])
    print ()
    
    if not args.ignore_runfile:
        print ("*** Fairness metrics of the TREC run file %s ***" % args.runfile)
        _metric_res = _fairr_metric.calc_FaiRR_retrievalresults(_retrivalresults)
        _ms = list(_metric_res['metrics_avg'].keys())
        _ms.sort()
        if args.print_qry_results:
            for _m in _ms:
                _cutoffs = list(_metric_res['metrics_perq'][_m].keys())
                _cutoffs.sort()
                for _cutoff in _cutoffs:
                    _qrys = list(_metric_res['metrics_perq'][_m][_cutoff].keys())
                    _qrys.sort()
                    for _qry in _qrys:
                        print ("%s_%d %d:" % (_m, _cutoff, _qry), _metric_res['metrics_perq'][_m][_cutoff][_qry])

        for _m in _ms:
            _cutoffs = list(_metric_res['metrics_avg'][_m].keys())
            _cutoffs.sort()
            for _cutoff in _cutoffs:
                print ("%s_%d All:" % (_m, _cutoff), _metric_res['metrics_avg'][_m][_cutoff])
        
    
    