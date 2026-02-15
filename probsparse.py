import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
beta = 0.15
gamma = 0.25
from utils.masking import TriangularCausalMask, ProbMask
import torch
from typing import Optional
from quant import Quantizer, fake_quantize_quarter_E5M2, fake_quantize_quarter_E4M3, quantize_tensor, quantize_tensor_channel_group
from qLinearLayer import QLinearLayer

#args.abits = 8
#args.act_group_size = 0        # per-channel AQ
#args.keeper = K                # e.g. 4–16
#args.keeper_precision = 2      # FP4 E4M3 or INT8
#reorder_index = None           # disable reordering

class InferenceModule(torch.nn.Module):
    def inference(self):
        for mod in self.modules():
            if mod != self:
                mod.inference()

class Value(InferenceModule):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        self.fc1 = QLinearLayer(self.fc1, args)

    def forward(self, x):
        return self.fc1(x)

class Key(InferenceModule):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = QLinearLayer(self.fc1, args)

    def forward(self, x):
        return self.fc1(x)

class Query(InferenceModule):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = QLinearLayer(self.fc1, args)

    def forward(self, x):
        return self.fc1(x)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.act_quant = Quantizer(args=args)
        
    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)  #32x8x96x64 , 25, 25
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape #32x8x96x64
        _, _, L_Q, _ = Q.shape #96

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E) #32x8x96x96x64
        index_sample = torch.randint(L_K, (L_Q, sample_k), device= 'cuda') # real U = U_part(factor*ln(L_k))*L_q #96x25
        K_sample = K_expand[:, :, torch.arange(L_Q, device ='cuda').unsqueeze(1), index_sample, :] #32x8x96x25x64
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2) #32x8x96x1x64 x 32x8x96x64x25 = 32x8x96x25

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K) #32x8x96x1
        M_top = M.topk(n_top, sorted=False)[1] #32x8x25

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q) 32x8x25x64
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k 32x8x25x64 x 32x8x64x96 = 32x8x25x96

        return Q_K, M_top #32x8x25x96, 32x8x25

    def _get_initial_context(self, V, L_Q):  # 32x8x96x64, 96
        B, H, L_V, D = V.shape  # 32x8x96x64
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)  # 32x8x1x64
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # 32x8x96x64 cloning therefore 96 repeated vectors
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex  # 32x8x96x64

    def _update_context(self, context_in, V, scores, index, L_Q,
                        attn_mask):  # 32x8x96x64, 32x8x96x64, #32x8x25x96, 2x8x25, 25, 0
        B, H, L_V, D = V.shape  # 32x8x96x64
        assert index.dim() == 3  # [B, H, n_top]
        index = index.long().to(V.device)

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores) #32x8x25x96
        # context_in[torch.arange(B)[:, None, None],
        # torch.arange(H)[None, :, None],
        # index, :]
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)  # 32x8x25x96 x 32x8x96x64 --> 32x8x25x64
        # therefore only 25 tensors are updated out of original 96
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)  # 32x8x25x64, #32x8x25x96
        else:
            return (context_in, None)  # 32x8x25x64

    def forward(self, queries, keys, values, attn_mask):  # 32x96x8x64
        B, L_Q, H, D = queries.shape  # 32, 96, 8, 64
        _, L_K, _, _ = keys.shape  # 96

        queries = queries.transpose(2, 1)  # 32x8x96x64
        keys = keys.transpose(2, 1)  # 32x8x96x64
        values = values.transpose(2, 1)  # 32x8x96x64
        keys = self.k_quant(keys)
        values = self.v_quant(values)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k) 25
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q) 25

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)  # 32x8x96x64, 25, 25 --> #32x8x25x96, 32x8x25
        a = scores_top/torch.sqrt(torch.tensor(queries.shape[-1]).float())
        dist_mask_tile = get_dist_mask_tile(a.size(1))
        a = a + 1 * dist_mask_tile.cuda()
        dir_mask = torch.triu(torch.ones((a.size(1), a.size(1))), diagonal=1).bool().cuda()
        a = a.masked_fill(dir_mask, float('-inf')).cuda()
        #a = F.softmax(a, dim=-1).cuda()

        # add scale factor
        scale = 1.0 / math.sqrt(D)        # 64 = 0.125
        if scale is not None:
            scores_top = scores_top * scale  # 32x8x25x96
            # get the context
            context = self._get_initial_context(values, L_Q)  # 32x8x96x64, 96 --> #32x8x96x64
            # update the context with selected top_k queries
            context, attn = self._update_context(context, values, a, index, L_Q, attn_mask)  # 32x8x25x64, #32x8x25x96

            return context.transpose(2, 1).contiguous(), attn  # 32x8x64x25, #32x8x25x96

# class QuerySelector(nn.Module):
#     def __init__(self, fraction=0.33):
#         super(QuerySelector, self).__init__()
#         self.fraction = fraction
#
#     def forward(self, queries, keys, values):
#         B, L_Q, D = queries.shape
#         _, L_K, _ = keys.shape
#         l_Q = int((1.0 - self.fraction) * L_Q)
#         K_reduce = torch.mean(keys.topk(l_Q, dim=1).values, dim=1).unsqueeze(1)
#         sqk = torch.matmul(K_reduce, queries.transpose(1, 2))
#         indices = sqk.topk(l_Q, dim=-1).indices.squeeze(1)
#         Q_sample = queries[torch.arange(B)[:, None], indices, :]  # factor*ln(L_q)
#         Q_K = torch.matmul(Q_sample, keys.transpose(-2, -1))
#         attn = torch.softmax(Q_K / math.sqrt(D), dim=-1)
#         mean_values = values.mean(dim=-2)
#         result = mean_values.unsqueeze(-2).expand(B, L_Q, mean_values.shape[-1]).clone()
#         result[torch.arange(B)[:, None], indices, :] = torch.matmul(attn, values).type_as(result)
#         return result, None

    # def inference(self):
    #     pass  # no parameters

def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(-2, -1))  # (B, L, L)
    m = m / math.sqrt(Q.shape[-1])
    return torch.softmax(m, -1)

def get_dist_mask_tile(sentence_len):
    row, col = torch.meshgrid(torch.arange(sentence_len), torch.arange(sentence_len))
    dis_mask = (row - col).abs()
    dis_mask = 1 / (1 + beta * np.power(dis_mask, 2))
    dis_mask = np.clip(dis_mask, gamma, 1.0)
    return dis_mask

def attention(Q, K, V):
    a = a_norm(Q, K).cuda()
    dist_mask_tile = get_dist_mask_tile(a.size(1)).cuda()
    a = a + 1 * dist_mask_tile.cuda()
    dir_mask = torch.triu(torch.ones((a.size(1), a.size(1))), diagonal=1).bool().cuda()
    a = a.masked_fill(dir_mask, float('-inf')).cuda()
    a = F.softmax(a, dim=-1).cuda()
    return torch.matmul(a, V)

class Attention(InferenceModule):
    def __init__(self, dim_val, dim_attn, debug=False, attn_type='prob'):
        super(Attention, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
        self.debug = debug
        self.qk_record = None
        self.qkv_record = None
        self.n = 0
        originalAttn: ProbAttention
        self.abits = args.abits
        self.num_key_value_heads = originalAttn.num_key_value_heads
        self.num_key_value_groups = originalAttn.num_key_value_groups

        
        self.act_quant = Quantizer(args=args)
        self.v_quant = Quantizer(args=args)
        self.k_quant = Quantizer(args=args) 
        self.register_buffer("reorder_index", None)

        if attn_type == "full":
            self.attentionLayer = None
        elif attn_type.startswith("prob"):
            self.attentionLayer = ProbAttention(mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False)
        else:
            raise Exception
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
            return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x, kv=None):
        if kv is None:
            if self.attentionLayer:
                qkv = self.attentionLayer(self.query(x), self.key(x), self.value(x))[0]
            else:
                qkv = attention(self.query(x), self.key(x), self.value(x))
                qkv = self.act_quant(qkv)
                return qkv
        return attention(self.query(x), self.key(kv), self.value(kv))

class Encoder(InferenceModule):
    def __init__(self, dim_val, dim_attn, n_heads=1, attn_type='prob'):
        super(Encoder, self).__init__()
        self.attn = Truncated_Cauchy_self_attention_layer(dim_val, dim_attn, n_heads, attn_type=attn_type)

        self.fc1 = Linear(dim_val, dim_val)
        self.fc2 = Linear(dim_val, dim_val)

        self.norm1 = LayerNorm(dim_val)
        self.norm2 = LayerNorm(dim_val)

        self.act_quant = Quantizer(args=args)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)

        a = self.fc1(F.elu(self.fc2(x)))
        a = self.act_quant(a)
        x = self.norm2(x + a)

        return x

    def record(self):
        self.attn.record()

class InferenceModuleList(torch.nn.ModuleList):
    def inference(self):
        for mod in self.modules():
            if mod != self:
                mod.inference()

class Truncated_Cauchy_self_attention_layer(InferenceModule):
    def __init__(self, dim_val, dim_attn, n_heads, attn_type):
        super(Truncated_Cauchy_self_attention_layer, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(Attention(dim_val, dim_attn, attn_type=attn_type))

        self.heads = InferenceModuleList(self.heads)
        self.fc = Linear(n_heads * dim_val, dim_val, bias=False)

    def forward(self, x, kv=None):
        a = []
        # for h in self.heads:
        #     a.append(h(x, kv=kv))
        for h in self.heads:
            head_out = h(x, kv=kv)
            if isinstance(head_out, tuple):
                head_out = head_out[0]
            a.append(head_out)

        a = torch.stack(a, dim=-1)  # combine heads
        a = a.flatten(start_dim=2)  # flatten all head outputs

        x = self.fc(a)

        return x

    def record(self):
        for h in self.heads:
            h.record()

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        if self.training:
            return super(LayerNorm, self).forward(x)
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight.data, self.bias.data, self.eps)

    def inference(self):
        self.training = False

class Linear(nn.Linear):
    def forward(self, x):
        if self.training:
            return super(Linear, self).forward(x)
        else:
            return F.linear(x, self.weight.data, self.bias.data if self.bias is not None else None)

    def inference(self):
        self.training = False

class Dropout(nn.Dropout):
    def forward(self, x=False):
        if self.training:
            return super(Dropout, self).forward(x)
        else:
            return x

    def inference(self):
        self.training = False

class PositionalEncoding(InferenceModule):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x

class InferenceModule(torch.nn.Module):
    def inference(self):
        for mod in self.modules():
            if mod != self:
                mod.inference()

class Truncated_Cauchy_self_attention_block(InferenceModule):
    def __init__(self, dim_val, dim_attn, input_size, out_seq_len, n_encoder_layers=1,
                 enc_attn_type='full', n_heads=1, dropout=0.1, debug=False):
        super(Truncated_Cauchy_self_attention_block, self).__init__()
        self._linear = nn.Linear(dim_val, 1)
        # Initiate encoder and Decoder layers
        self.encs = []
        for i in range(n_encoder_layers):
            self.encs.append(Encoder(dim_val, dim_attn, n_heads, attn_type=enc_attn_type))
        self.encs = InferenceModuleList(self.encs)
        self.decs = []
        self.pos = PositionalEncoding(dim_val)
        self.enc_dropout = Dropout(dropout)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = Linear(input_size, dim_val)
        # print("dim_val")
        # print(dim_val)
        self.dec_input_fc = Linear(input_size, dim_val)
        self.debug = debug

    def forward(self, x):
        # encoder
        x = x.to(torch.float32)
        a = self.enc_input_fc(x)
        b = self.enc_dropout(a)
        c = self.pos(b)
        e = self.encs[0](c)
        for enc in self.encs[1:]:
            e = e.cuda()
            e = enc(e)
        if self.debug:
            print('Encoder output size: {}'.format(e.shape))
        return e

    # def record(self):
    #     self.debug = True
    #     for enc in self.encs:
    #         enc.record()
    #     for dec in self.decs:
    #         dec.record()