import numpy as np
from typing import Dict, List, Tuple, Optional, overload
import pdb

import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn.modules.linear import Linear

from allennlp.models import Model
from allennlp.nn import util

from transformers import BertModel


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        grad_input = -grad_output*ctx.alpha
        return grad_input, None


class AdvBert(Model):
    def __init__(self,
                 bert: BertModel,
                 adv_rev_factor = 1.0,
                 cls_token_id=101, 
                 sep_token_id=102):
        super(AdvBert, self).__init__(vocab=None) # (?)
        
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_input_length = 512
        
        self._bert = bert
        self._embedding_size = self._bert.config.hidden_size 
        self._output_projection_layer = Linear(self._embedding_size, 2)
        self.adv_rev_factor = adv_rev_factor
        
        self.adversary_net = torch.nn.Sequential(
            torch.nn.Linear(self._embedding_size, self._embedding_size, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(self._embedding_size, 2, bias=True))
        
        
    def forward(self, query: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
        tok_seq, pad_mask, seg_mask = self.prepare_batch(query, document)
        
        out = self._bert(input_ids=tok_seq, attention_mask=pad_mask, token_type_ids=seg_mask)
        cls_output = out[0][:,0,:]
        
        scores = self._output_projection_layer(cls_output)
        lprobs = torch.nn.LogSoftmax(dim=-1)(scores)
        rels = lprobs[:,0]
        
        adversary_scores = self.adversary_net.forward(ReverseLayerF.apply(cls_output, self.adv_rev_factor))
        adversary_lprobs = torch.nn.LogSoftmax(dim=-1)(adversary_scores)
        
        return {"rels" : rels, "logprobs": lprobs, "adv_logprobs": adversary_lprobs}

    def prepare_batch(self, query: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
        
        bsz = document.size(0)
        
        tok_seq = torch.full((bsz, self.max_input_length), 0, dtype=int).cuda()
        seg_mask = torch.full((bsz, self.max_input_length), 0, dtype=int).cuda()
        seg_1_value = 0
        seg_2_value = 1
        
        tok_seq[:, 0] = torch.full((bsz, 1), self.cls_token_id, dtype=int)[:, 0]
        seg_mask[:, 0] = torch.full((bsz, 1), seg_1_value, dtype=int)[:, 0]
        
        for batch_i in range(bsz):
            # query
            _offset = 1
            _vec = query[batch_i]
            _length = len(_vec[_vec != 0])
            tok_seq[batch_i, _offset:_length+_offset] = query[batch_i, :_length]
            seg_mask[batch_i, _offset:_length+_offset] = torch.full((_length, 1), seg_1_value, dtype=int)[:, 0]
            _offset += _length
            
            tok_seq[batch_i, _offset:_offset+1] = self.sep_token_id
            seg_mask[batch_i, _offset:_offset+1] = seg_1_value
            _offset += 1
            
            # document
            ## we assume that length of query (+2) never exceeds <max_input_length>
            ## therefore we only truncate the document
            ## in extreme cases this can hurt
            _vec = document[batch_i]
            _length = len(_vec[_vec != 0])
            _fill_until = _length + _offset
            if _fill_until >= self.max_input_length:
                _fill_until = self.max_input_length - 1 # leaving space for the last <sep> 
                _length = _fill_until - _offset
            tok_seq[batch_i, _offset:_fill_until] = document[batch_i, :_length]
            seg_mask[batch_i, _offset:_fill_until] = torch.full((_length, 1), seg_2_value, dtype=int)[:, 0]
            _offset += _length
            
            tok_seq[batch_i, _offset:_offset+1] = self.sep_token_id
            seg_mask[batch_i, _offset:_offset+1] = seg_2_value
            _offset += 1
            
        pad_mask = util.get_text_field_mask({"tokens":tok_seq}).cuda()
        
        return tok_seq, pad_mask, seg_mask
