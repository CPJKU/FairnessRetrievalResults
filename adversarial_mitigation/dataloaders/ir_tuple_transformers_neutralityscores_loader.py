# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py

from typing import Dict, List, Set
from typing import Callable
import logging
import sys
import numpy as np

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField, ArrayField
#from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from allennlp.data.instance import Instance

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class IrTupleTransformersNeutralityScoresDatasetReader(DatasetReader):
    def __init__(self,
                 transformers_tokenizer: PreTrainedTokenizer,
                 add_special_tokens: bool = False,
                 #source_add_bos_token: bool = True,
                 max_doc_length:int = -1,
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
        try:
            with open(cached_path(file_path), "r", encoding="utf8") as data_file:
                #logger.info("Reading instances from lines in file at: %s" % file_path)
                for line_num, line in enumerate(data_file):
                    line = line.strip("\n")

                    if not line:
                        continue

                    line_parts = line.split('\t')
                    if len(line_parts) != 6:
                        sys.stdout.write ("Invalid line format: %s (line number %d)\n" % (line, line_num + 1))
                        sys.stdout.flush()
                        raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                    query_id, doc_id, query_sequence, doc_sequence, query_neutscore, doc_neutscore = line_parts
                    if self._preprocess != None:
                        query_sequence = self._preprocess(query_sequence)
                        doc_sequence = self._preprocess(doc_sequence)
                    query_neutscore = float(query_neutscore)
                    doc_neutscore = float(doc_neutscore)
                    yield self.text_to_instance(query_id, doc_id, query_sequence, doc_sequence, query_neutscore, doc_neutscore)
        except Exception as e: 
            sys.stdout.write(e)
            sys.stdout.flush()

    @overrides
    def text_to_instance(self, query_id:str, doc_id:str, query_sequence: str, doc_sequence: str, 
                         query_neutscore: float, doc_neutscore: float) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        query_id_field = MetadataField(int(query_id))
        doc_id_field = MetadataField(doc_id)
        
        # dummy code to prevent empty queries
        if len(query_sequence.strip()) == 0:
            query_sequence = "@@UNKNOWN@@"

        query_pre_tokenized = query_sequence.split()
        if self._filter_gendered_tokens:
            query_pre_tokenized = [_t for _t in query_pre_tokenized if _t.lower() not in self._gendered_tokens]

        if self._max_query_length > -1:
            query_pre_tokenized = query_pre_tokenized[:self._max_query_length]
        query_tokenized = self._transformers_tokenizer(' '.join(query_pre_tokenized),
                                                       truncation = True,
                                                       add_special_tokens = self._add_special_tokens)["input_ids"]
        
        doc_pre_tokenized = doc_sequence.split()
        if self._filter_gendered_tokens:
            doc_pre_tokenized = [_t for _t in doc_pre_tokenized if _t.lower() not in self._gendered_tokens]

        if self._max_doc_length > -1:
            doc_pre_tokenized = doc_pre_tokenized[:self._max_doc_length]
        doc_tokenized = self._transformers_tokenizer(' '.join(doc_pre_tokenized),
                                                     truncation = True,
                                                     add_special_tokens = self._add_special_tokens)["input_ids"]

        query_field = ArrayField(np.array(query_tokenized))
        doc_field = ArrayField(np.array(doc_tokenized))

        ## labels of protected attributes, computed from neutrality scores
        protected_label = 1 if (query_neutscore < 1) or (doc_neutscore < 1) else 0
        protected_label_field = LabelField(protected_label, skip_indexing=True)
        
        return Instance({
            "query_id" : query_id_field,
            "doc_id" : doc_id_field,
            "query_tokens" : query_field,
            "doc_tokens" : doc_field,
            "protected_label" : protected_label_field,
        })


