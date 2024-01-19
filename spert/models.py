import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel, BertTokenizer
from transformers import AlbertPreTrainedModel, AlbertModel, AlbertConfig, AlbertTokenizer

from spert import sampling
from spert import util
import math
import random
from collections import OrderedDict
import json
import numpy as np
from torch.autograd import Variable
from math import sqrt


from typing import Dict, Optional
import torch.nn.functional as F
from torch import Tensor

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """  
    emb_size = h.shape[-1]                  #768

    token_h = h.view(-1, emb_size)  
    flat = x.contiguous().view(-1)  
    
    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]  

    return token_h  

def del_tensor_ele_n(arr, index, n):
    """
    arr: 输入tensor
    index: 需要删除位置的索引
    n: 从index开始，需要删除的行数
    """
    arr1 = arr[:,0:index,:]   
    arr2 = arr[:,index+n:,:]  
    return torch.cat((arr1,arr2),dim=1)


def getPosEncodingMatrix(max_len,d_emb):
    # pos_enc = np.array([[pos/np.power(10000,2*(j//2)/d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in range(max_len)])
    P = torch.zeros((1, max_len, d_emb))
    pos_enc = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(256, torch.arange(0, d_emb, 2, dtype=torch.float32) / d_emb)
    # pos_enc[1:,0::2] = np.sin(pos_enc[1:,0::2])
    # pos_enc[1:,1::2] = np.cos(pos_enc[1:,1::2])
    P[:, :,0::2] = np.sin(pos_enc)
    P[:, :,1::2] = np.cos(pos_enc)

    return P.to(_device)

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(256, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)



# def sequence_mask(X, valid_len, value=0):
#     maxlen = X.size(1)
#     mask = torch.arange((maxlen), dtype=torch.float32,device=X.device)[None, :] < valid_len[:, None].to(_device)
#     X[~mask] = value
#     return X


# def masked_softmax(X, valid_lens):
#     """通过在最后一个轴上掩蔽元素来执行softmax操作"""
#     # X:3D张量，valid_lens:1D或2D张量
#     if valid_lens is None:
#         return nn.functional.softmax(X, dim=-1)
#     else:
#         shape = X.shape
#         if valid_lens.dim() == 1:
#             valid_lens = torch.repeat_interleave(valid_lens, shape[1])
#         else:
#             valid_lens = valid_lens.reshape(-1)
#         # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
#         X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
#                               value=-1e6)
#         return nn.functional.softmax(X.reshape(shape), dim=-1)

'''attention'''
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)   
        # self.attention_weights = masked_softmax(scores, valid_lens)
        scores[scores==0] = -1e+30
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)

'''multihead attention'''
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
       
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias).to(_device)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias).to(_device)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias).to(_device)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias).to(_device)

    def forward(self, queries, keys, values):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # if valid_lens is not None:
        #     # 在轴0，将第一项（标量或者矢量）复制num_heads次，
        #     # 然后如此复制第二项，然后诸如此类。
        #     valid_lens = torch.repeat_interleave(
        #         valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

def get_relation_embedding(batch_size, relation_types, embedding):
    relation_list = []
    for i in range(relation_types):
        relation_list.append(i)
    return embedding(torch.tensor(relation_list).to(_device)).repeat(batch_size, 1, 1)

class CrossMultiAttention(nn.Module):
    def __init__(self, dim, num_heads=6, attn_drop=0.2, proj_drop=0.2, qkv_bias=False, qk_scale=None):
        super().__init__()

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.wq = nn.Linear(dim, dim, bias=qkv_bias).to(_device)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias).to(_device)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias).to(_device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.addnorm = AddNorm([1, dim])

    def forward(self, y, x_cls):

        y_all = torch.concat((y, x_cls), dim=1)
        B, N, C = y_all.shape   
        
        q = self.wq(x_cls).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
        k = self.wk(y_all).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(y_all).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale   
        attn = attn.softmax(dim=-1)
        
        # print(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  
        x = self.proj(x)
        x = self.proj_drop(x)
        o = x + x_cls  

        return self.addnorm(o)
        #return x

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, Y):
        return self.ln(Y)

 
 
class MLPAttentionNetwork(nn.Module):
 
    def __init__(self, hidden_dim, src_length_masking=True):
        super(MLPAttentionNetwork, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.src_length_masking = src_length_masking
 
        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # v.T
        self.proj_v = nn.Linear(self.hidden_dim, 1, bias=False)
 
    def forward(self, x):
        """
        :param x: seq_len * batch_size * hidden_dim
        :param x_lengths: batch_size
        :return: batch_size * hidden_dim
        """
        mlp_x = self.proj_w(x)
        att_scores = self.proj_v(mlp_x)
        att_scores[att_scores==0] =-1e30
        normalized_masked_att_scores = nn.functional.softmax(att_scores, dim=-1)
        attn_x = normalized_masked_att_scores * x
 
        return attn_x

class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
 
        #定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)
 
    def forward(self, x):
        # x: batch, n, dim_q
        #根据文本获得相应的维度
 
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q
 
        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)

        #q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        #归一化获得attention的相关系数s
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        #attention系数和v相乘，获得最终的得分
        return torch.bmm(dist, v)

def cumsoftmax(x):
    return torch.cumsum(F.softmax(x,-1),dim=-1)

class SpERT(BertPreTrainedModel):

# class SpERT(AlbertPreTrainedModel):

    """ Span-based model to jointly extract entities and relations """

    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, head: int, freeze_transformer: bool, max_pairs: int = 100):
        super(SpERT, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)

        # layers
        self.linear_r = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.linear_s = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.linear_o = nn.Linear(config.hidden_size, config.hidden_size, bias=False) 

        self.cross_attention_s = CrossMultiAttention(config.hidden_size, head)
        
        self.cross_attention_o = CrossMultiAttention(config.hidden_size, head)  


        # self.rel_classifier = nn.Linear(config.hidden_size * 4 + size_embedding*2, relation_types)
        self.rel_classifier = nn.Linear(config.hidden_size * (1+1 + 1+2) + size_embedding*2, relation_types)  
        
        self.entity_classifier_s = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types-1)  

        self.entity_classifier_o = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types-1)  

        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False


    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        # h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]  

        batch_size = encodings.shape[0]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)    # embed entity candidate sizes  
       
        entity_clf, e_r, cross_cls = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        entity_o_clf, e_o_r, cross_o_cls = self._classify_entities_o(encodings, h, entity_masks, size_embeddings)  


        
        # classify relations      
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)  
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)  



        index_judge_0, index_judge_1 = self._index_judge(entity_clf, entity_o_clf)  
        
        
        


        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):  
            # classify relation candidates
            
            # chunk_rel_logits = self._classify_relations(e_r, cross_cls, size_embeddings,
            #                                             relations, rel_masks, h_large, h, i)            
            chunk_rel_logits = self._classify_relations(index_judge_0, e_r, cross_cls, index_judge_1, e_o_r, cross_o_cls, 
                                                        size_embeddings, relations, rel_masks, h_large, h, i)
            
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits   

        
        # return entity_clf, rel_clf  
        
        
        #  

        
        # entity_clf_mixed = entity_clf * index_judge_0 + entity_o_clf * index_judge_1   
        

        return entity_clf, entity_o_clf, rel_clf

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]
        # outputs = self.bert(input_ids=encodings, attention_mask=context_masks, output_attentions=True)
        # h = outputs.last_hidden_state
        # attention = outputs.attentions
        # print(1111111)
        # print(attention[0][0][0].mean(0))

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes

        entity_clf, e_r, cross_cls = self._classify_entities(encodings, h, entity_masks, size_embeddings)
        entity_o_clf, e_o_r, cross_o_cls = self._classify_entities_o(encodings, h, entity_masks, size_embeddings)        

        
        # e_r -> Tensor( Batch_size , pos+neg候选实体数量 , hidden_size 768 )    ;    
        # cross_cls ->  Tensor ( Batch_size , 1 , hidden_size 768 )

        


        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        
        # relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
        #                                                             entity_sample_masks, ctx_size)

        # relations_o, rel_masks_o, rel_sample_masks_o = self._filter_spans(entity_o_clf, entity_spans,
        #                                                             entity_sample_masks, ctx_size)

        
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_o_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)
        

        

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)
        


        index_judge_0, index_judge_1 = self._index_judge(entity_clf, entity_o_clf)
        
        
        


        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(index_judge_0, e_r, cross_cls, index_judge_1, e_o_r, cross_o_cls, size_embeddings,
                                                        relations, rel_masks, h_large, h, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        
        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)
        entity_o_clf = torch.softmax(entity_o_clf, dim=2)  


        
        
        # entity_clf_mixed = entity_clf * index_judge_0 + entity_o_clf * index_judge_1

        

        entity_clf_mixed = torch.zeros([entity_clf.shape[0], entity_clf.shape[1], 3]).to(_device)
        
        entity_clf_mixed[:, :, 1] += (index_judge_0 * entity_clf)[:, :, 1]
        entity_clf_mixed[:, :, 0] += (index_judge_0 * entity_clf)[:, :, 0]
        
        entity_clf_mixed[:, :, 2] += (index_judge_1 * entity_o_clf)[:, :, 1]
        entity_clf_mixed[:, :, 0] += (index_judge_1 * entity_o_clf)[:, :, 0]
        

        return entity_clf_mixed, rel_clf, relations



    def _index_judge(self, entity_clf, entity_o_clf):  
        
        
        
        
        
        
        
        '''
                              |    0  Aspect classify        /\
                              |---------------------------- //\\                           
                              | 0 No Entity  | 1 Aspect      ||
        ----------------------|--------------|-------------
            1   | 0 No Entity |     max      |   Aspect 0
        Opinion |-------------|--------------|-------------  
        classify| 1 Opinion   |   Opinion 1  |   max \ (可以尝试默认用Aspect或者Opinion)
        ---------------------------------------------------           
               <<== 
              
        '''
        
        max_entity_s = entity_clf.max(dim=2)  
        max_entity_o = entity_o_clf.max(dim=2)        
        
        # index_judge_s = (max_entity_s[0] > max_entity_o[0]).bool()   
        index_judge_o = (max_entity_s[1] < max_entity_o[1]).int()   
        
        index_judge_equ_mask = (max_entity_s[1] == max_entity_o[1]).int()    
        
        index_judge_max = (max_entity_s[0] < max_entity_o[0]).int()  
        
        index_judge_1 = index_judge_o + index_judge_equ_mask * index_judge_max   # Tensor( Batch_size , 正负实体数量 )  #  * 逐元素矩阵乘法  
        
        
        index_judge_1 = index_judge_1.unsqueeze(dim=-1)   # Tensor( Batch_size , 正负实体数量 , 1 )  
        
        index_judge_0 = 1 - index_judge_1

        
        return index_judge_0, index_judge_1


    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        #get inicial cls token
        
        h_r_all = self.linear_r(h)  
        h_s_all = self.linear_s(h)  


        h_r_cls = get_token(h_r_all, encodings, self._cls_token).unsqueeze(dim=1)  
        h_s_cls = get_token(h_s_all, encodings, self._cls_token).unsqueeze(dim=1)  
        
        h_r = del_tensor_ele_n(h_r_all, 0, 1)  
        h_s = del_tensor_ele_n(h_s_all, 0, 1)  

        # cross_cls = self.cross_attention_r(h_r, h_s_cls) + self.cross_attention_r(h_s, h_r_cls)

        cross_cls_r = self.cross_attention_s(h_s, h_r_cls)  
        cross_cls_s = self.cross_attention_s(h_r, h_s_cls)  


        # cross_cls_r1 = self.cross_attention(h_s, cross_cls_r)
        # cross_cls_s1 = self.cross_attention(h_r, cross_cls_s)


        #get entity candidate spans    
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)  
        
        entity_spans = m + h_s_all.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        # relation_spans = m + h_r_all.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        
        
        # max pool entity candidate spans
        entity_spans_pool = entity_spans.max(dim=2)[0]      
        # relation_spans_pool = relation_spans.max(dim=2)[0]

        cross_cls_mixed = cross_cls_r + cross_cls_s

        ner_ctx = cross_cls_mixed.repeat(1, entity_spans_pool.shape[1], 1)     
        # ner_ctx = cross_cls_s.repeat(1, entity_spans_pool.shape[1], 1)     
        
        entity_repr = torch.cat([ner_ctx, entity_spans_pool, size_embeddings], dim=2)    
        #entity_repr = torch.cat([entity_spans_pool, size_embeddings], dim=2)   
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier_s(entity_repr)

        # cross_cls_s_r = cross_cls_s + cross_cls_r

        # return entity_clf, relation_spans_pool, h_r_all
        # return entity_clf, relation_spans_pool, cross_cls_mixed
        return entity_clf, entity_spans_pool, cross_cls_mixed


    
    def _classify_entities_o(self, encodings, h, entity_masks, size_embeddings):
        #get inicial cls token
        
        h_r_all = self.linear_r(h)  
        h_o_all = self.linear_o(h)  


        h_r_cls = get_token(h_r_all, encodings, self._cls_token).unsqueeze(dim=1)  
        h_o_cls = get_token(h_o_all, encodings, self._cls_token).unsqueeze(dim=1)  
        
        h_r = del_tensor_ele_n(h_r_all, 0, 1)  
        h_o = del_tensor_ele_n(h_o_all, 0, 1)  

        # cross_cls = self.cross_attention_r(h_r, h_s_cls) + self.cross_attention_r(h_s, h_r_cls)

        cross_cls_r = self.cross_attention_o(h_o, h_r_cls)  
        cross_cls_o = self.cross_attention_o(h_r, h_o_cls)  


        # cross_cls_r1 = self.cross_attention(h_s, cross_cls_r)
        # cross_cls_s1 = self.cross_attention(h_r, cross_cls_s)


        #get entity candidate spans    
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)  
        
        
        
        entity_spans = m + h_o_all.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        # relation_spans = m + h_r_all.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        
        
        # max pool entity candidate spans
        entity_spans_pool = entity_spans.max(dim=2)[0]      
        # relation_spans_pool = relation_spans.max(dim=2)[0]

        cross_cls_mixed = cross_cls_r + cross_cls_o

        ner_ctx = cross_cls_mixed.repeat(1, entity_spans_pool.shape[1], 1)     
        # ner_ctx = cross_cls_o.repeat(1, entity_spans_pool.shape[1], 1)     

        entity_repr = torch.cat([ner_ctx, entity_spans_pool, size_embeddings], dim=2)
        #entity_repr = torch.cat([entity_spans_pool, size_embeddings], dim=2)   
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier_o(entity_repr)        

        # cross_cls_r_o = cross_cls_o + cross_cls_r

        # return entity_clf, relation_spans_pool, h_r_all
        # return entity_clf, relation_spans_pool, cross_cls_r, cross_cls_o
        return entity_clf, entity_spans_pool, cross_cls_mixed

    
    def _classify_relations(self, index_judge_0, e_r, cross_cls, index_judge_1, e_o_r, cross_o_cls, size_embeddings, relations, rel_masks, h_large, h, chunk_start):
        batch_size = relations.shape[0]   

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h_large = h_large[:, :relations.shape[1], :]

       
        
        # 
        # 
        # 
        
        # 
        # 
        # '''
        #                       |    0  Aspect classify        /\
        #                       |---------------------------- //\\                           
        #                       | 0 No Entity  | 1 Aspect      ||
        # ----------------------|--------------|-------------
        #     1   | 0 No Entity |     max      |   Aspect 0
        # Opinion |-------------|--------------|-------------  
        # classify| 1 Opinion   |   Opinion 1  |   max \ (可以尝试默认用Aspect或者Opinion)
        # ---------------------------------------------------           
        #        <<==               
        # '''
        entity_pairs_mixed = torch.stack([(e_r[i] * index_judge_0[i])[relations[i]] + (e_o_r[i] * index_judge_1[i])[relations[i]] for i in range(relations.shape[0])])  
        entity_pairs_mixed = entity_pairs_mixed.view(batch_size, entity_pairs_mixed.shape[1], -1)
    



        # get corresponding size embeddings   
        size_pair_embeddings = util.batch_index(size_embeddings, relations)  
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens              
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)  
        rel_ctx = m + h_large    
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]  
        # set the context vector of neighboring or adjacent entity candidates to zero  
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0   


        '''
                            |      Aspect classify
                            |---------------------------                            
                            |  No Entity  |   Aspect
        --------------------|-------------|-------------
        Opinion | No Entity |    throw    |   Aspect
        classify|-----------|-------------|-------------  
                | Opinion   |   Opinion   |   max \ (可以尝试默认用Aspect或者Opinion)
        ------------------------------------------------           

        '''

        rel_cls = cross_cls.repeat(1, size_pair_embeddings.shape[1], 1)  

        # rel_cls_s = cross_cls_s.repeat(1, size_pair_embeddings.shape[1], 1)

        rel_o_cls = cross_o_cls.repeat(1, size_pair_embeddings.shape[1], 1)

        # rel_cls_o = cross_cls_o.repeat(1, size_pair_embeddings.shape[1], 1)


 
        rel_repr = torch.cat([rel_ctx, rel_cls, rel_o_cls, entity_pairs_mixed, size_pair_embeddings], dim=2)  


        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)   
        return chunk_rel_logits


    
    
    def _filter_spans(self, entity_clf, entity_o_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]   
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none) 的同时，去除负样本的实体 ->  * entity_sample_masks.long()
        
        

        entity_logits_max_o = entity_o_clf.argmax(dim=-1) * entity_sample_masks.long()


        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []


        
        
        

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)  
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()  
            non_zero_indices = non_zero_indices.tolist()

            
            non_zero_indices_o = (entity_logits_max_o[i] != 0).nonzero().view(-1)  
            non_zero_spans_o = entity_spans[i][non_zero_indices_o].tolist()  
            non_zero_indices_o = non_zero_indices_o.tolist()


            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices_o, non_zero_spans_o):
                    if i1 != i2:
                        rels.append((i1, i2))   
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))   
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))   
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # for i in range(batch_size):
        #     rels = []
        #     rel_masks = []
        #     sample_masks = []

        #     # get spans classified as entities
        #     non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)  
            
        #     non_zero_spans = entity_spans[i][non_zero_indices].tolist()  
        #     non_zero_indices = non_zero_indices.tolist()

        #     # create relations and masks
        #     for i1, s1 in zip(non_zero_indices, non_zero_spans):
        #         for i2, s2 in zip(non_zero_indices, non_zero_spans):
        #             if i1 != i2:
        #                 rels.append((i1, i2))
        #                 rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
        #                 sample_masks.append(1)

        #     if not rels:
        #         # case: no more than two spans classified as entities
        #         batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
        #         batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
        #         batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
        #     else:
        #         # case: more than two spans classified as entities
        #         batch_relations.append(torch.tensor(rels, dtype=torch.long))
        #         batch_rel_masks.append(torch.stack(rel_masks))
        #         batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)  

        return batch_relations, batch_rel_masks, batch_rel_sample_masks 
        
        
        

 
    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)


# Model access

_MODELS = {
    'spert': SpERT,
}


def get_model(name):
    return _MODELS[name]
