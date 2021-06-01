# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py

from typing import Dict, List, Set
from typing import Callable
import logging
import pdb
import numpy as np

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.fields import TextField, LabelField, ArrayField
#from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.instance import Instance

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

                                            
class IrTripleTransformersNeutralityScoresDatasetReader(DatasetReader):
    def __init__(self,
                 transformers_tokenizer: PreTrainedTokenizer,
                 add_special_tokens: bool = False,
                 max_doc_length: int = -1, # in words
                 max_query_length:int = -1,
                 lazy: bool = False,
                 preprocess: Callable = None,
                 filter_gendered_tokens: bool = False,
                 gendered_tokens: Set = []
                 ) -> None:
        super().__init__(lazy)
        #self._pre_tokenizer = WhitespaceTokenizer()
        self._transformers_tokenizer = transformers_tokenizer
        self._add_special_tokens = add_special_tokens
        self._max_doc_length = max_doc_length
        self._max_query_length = max_query_length
        self._preprocess = preprocess
        self._filter_gendered_tokens = filter_gendered_tokens
        self._gendered_tokens = gendered_tokens                 

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            #logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue
 
                line_parts = line.split('\t')
                if len(line_parts) != 6:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                query_sequence, doc_pos_sequence, doc_neg_sequence, query_neutscore, doc_pos_neutscore, doc_neg_neutscore = line_parts
                if self._preprocess != None:
                    query_sequence = self._preprocess(query_sequence)
                    doc_pos_sequence = self._preprocess(doc_pos_sequence)
                    doc_neg_sequence = self._preprocess(doc_neg_sequence)
                query_neutscore = float(query_neutscore)
                doc_pos_neutscore = float(doc_pos_neutscore)
                doc_neg_neutscore = float(doc_neg_neutscore)
                yield self.text_to_instance(query_sequence, doc_pos_sequence, doc_neg_sequence, 
                                            query_neutscore, doc_pos_neutscore, doc_neg_neutscore)

    @overrides
    def text_to_instance(self, query_sequence: str, doc_pos_sequence: str, doc_neg_sequence: str, 
                         query_neutscore: float, doc_pos_neutscore: float, doc_neg_neutscore: float) -> Instance:  # type: ignore

        if len(query_sequence.strip()) == 0:
            query_sequence = "@@UNKNOWN@@"
        
        query_pre_tokenized = query_sequence.split()
        if self._filter_gendered_tokens:
            query_pre_tokenized = [_t for _t in query_pre_tokenized if _t.lower() not in self._gendered_tokens]

        if self._max_query_length > -1: # in words
            query_pre_tokenized = query_pre_tokenized[:self._max_query_length]

        query_tokenized = self._transformers_tokenizer(' '.join(query_pre_tokenized),
                                                       truncation = True,
                                                       add_special_tokens = self._add_special_tokens)["input_ids"]
        
        doc_pos_pre_tokenized = doc_pos_sequence.split()
        if self._filter_gendered_tokens:
            doc_pos_pre_tokenized = [_t for _t in doc_pos_pre_tokenized if _t.lower() not in self._gendered_tokens]
            
        if self._max_doc_length > -1: # in words
            doc_pos_pre_tokenized = doc_pos_pre_tokenized[:self._max_doc_length]
            
        doc_pos_tokenized = self._transformers_tokenizer(' '.join(doc_pos_pre_tokenized),
                                                         truncation = True,
                                                         add_special_tokens = self._add_special_tokens)["input_ids"]
        
        doc_neg_pre_tokenized = doc_neg_sequence.split()
        if self._filter_gendered_tokens:
            doc_neg_pre_tokenized = [_t for _t in doc_neg_pre_tokenized if _t.lower() not in self._gendered_tokens]

        if self._max_doc_length > -1: # in words
            doc_neg_pre_tokenized = doc_neg_pre_tokenized[:self._max_doc_length]    
            
        doc_neg_tokenized = self._transformers_tokenizer(' '.join(doc_neg_pre_tokenized),
                                                         truncation = True, 
                                                         add_special_tokens = self._add_special_tokens)["input_ids"]

        query_field = ArrayField(np.array(query_tokenized))
        doc_pos_field = ArrayField(np.array(doc_pos_tokenized))
        doc_neg_field = ArrayField(np.array(doc_neg_tokenized))

        ## labels of protected attributes, computed from neutrality scores
        protected_label_pos = 1 if (query_neutscore < 1) or (doc_pos_neutscore < 1) else 0
        protected_label_neg = 1 if (query_neutscore < 1) or (doc_neg_neutscore < 1) else 0
        protected_label_pos_field = LabelField(protected_label_pos, skip_indexing=True)
        protected_label_neg_field = LabelField(protected_label_neg, skip_indexing=True)
        
        return Instance({
            "query_tokens" : query_field,
            "doc_pos_tokens" : doc_pos_field,
            "doc_neg_tokens" : doc_neg_field,
            "protected_label_pos" : protected_label_pos_field,
            "protected_label_neg" : protected_label_neg_field,
        })
