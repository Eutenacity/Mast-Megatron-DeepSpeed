import argparse
import torch
from torch import nn
from torch import distributed as dist
from torch.nn import functional as F
import math
import numpy as np

_groups = None
from torch import Tensor
import queue
def a2a_ffn_overlap_new(input,expert_fn,a2a_ffn_overlap_degree,group,encode='no'):
    split_dim=1
    C.AllToAllStatus.init(group, a2a_ffn_overlap_degree, split_dim)
    tutel_custom_kernel.clear_ptr_lst()

    split_size = input.shape[1] // a2a_ffn_overlap_degree
    input_split = list(input.split(split_size, dim=1))
    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = input_split[i].contiguous()
    for i in range(a2a_ffn_overlap_degree):

        input_split[i] = moe_utils.custom_compress(input_split[i])

        input_split[i] = moe_utils.custom_a2a(input_split[i] )
    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = moe_utils.custom_decompress(
            input_split[i]
        )
        input_split[i]=expert_fn(input_split[i])
        input_split[i] = moe_utils.custom_compress(input_split[i])

        input_split[i] = moe_utils.custom_a2a(input_split[i] )
    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = moe_utils.custom_decompress(
            input_split[i]
        )

    output = torch.cat(input_split, dim=1)

    return output
@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()
USE_EINSUM = True
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)
def _set_groups(**kwargs):
    global _groups
    _groups = kwargs


def get_groups():
    global _groups
    return _groups
def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)
def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator
def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def moe_init(args):
    # Create a comm prependicular to the pipeline group as gate group
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    for i in range(0, args.es_size):
        ranks = range(i, world_size, args.es_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            ep_group = group
    for i in range(0, world_size, args.es_size):
        ranks = range(i, i + args.es_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            es_group = group
    _set_groups(ep_group=ep_group, es_group=es_group)
class Attention(nn.Module):
    def __init__(self, args,mp_group) -> None:
        super().__init__()

        self.layer_norm=torch.nn.LayerNorm(args.M,eps=1e-5,elementwise_affine=True)
        self.hidden_size=args.M
        qkv_size_per_partition = (args.M// args.mp_size) * 3
        out_size_per_partition = args.M // args.mp_size
        
        if args.M>2048 and args.fp16:
            self.attn_qkvw = nn.Parameter(0.001*torch.randn(qkv_size_per_partition,args.M).half().cuda(),
                                      requires_grad=True)
        
            self.attn_qkvb = nn.Parameter(torch.zeros(qkv_size_per_partition).half().cuda(),
                                        requires_grad=True)
            
            self.attn_ow = nn.Parameter(0.001*torch.randn(args.M,
                                                    out_size_per_partition ).half().cuda(),
                                        requires_grad=True)

            self.attn_ob = nn.Parameter(torch.zeros(args.M).half().cuda(),
                                        requires_grad=True)
        else:
            self.attn_qkvw = nn.Parameter(0.001*torch.randn(qkv_size_per_partition,args.M).cuda(),
                                      requires_grad=True)
        
            self.attn_qkvb = nn.Parameter(torch.zeros(qkv_size_per_partition).cuda(),
                                        requires_grad=True)
            
            self.attn_ow = nn.Parameter(0.001*torch.randn(args.M,
                                                    out_size_per_partition ).cuda(),
                                        requires_grad=True)

            self.attn_ob = nn.Parameter(torch.zeros(args.M).cuda(),
                                        requires_grad=True)
        self.num_attention_heads_per_partition = args.heads // args.mp_size
        self.hidden_size_per_partition = args.M // args.mp_size
        self.hidden_size_per_attention_head = args.M // args.heads

        self.mp_group = mp_group
        self.norm_factor=math.sqrt(args.M// args.heads)
        self.attention_dropout = torch.nn.Dropout(0.2)
    def forward(self,input):
        seq=input.shape[1]
        if self.hidden_size>2048 and args.fp16:

            norm_input=F.layer_norm(input,(self.hidden_size,),self.layer_norm.weight.half(),self.layer_norm.bias.half())
        else:
            norm_input=self.layer_norm(input)
        
        qkv_out=F.linear(norm_input,self.attn_qkvw,self.attn_qkvb)
     
        new_tensor_shape = qkv_out.size()[:-1] + \
                (3,self.num_attention_heads_per_partition,self.hidden_size_per_attention_head)
        mixed_x_layer = qkv_out.view(*new_tensor_shape)
        mixed_x_layer=mixed_x_layer.permute(1,0,3,2,4).contiguous().view(new_tensor_shape[1],new_tensor_shape[0],new_tensor_shape[3],-1)
        
        (query_layer,
        key_layer,
        value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

    
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))
        
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)
        
        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=qkv_out.device)
        

        norm_factor=self.norm_factor
        # Raw attention scores. [b * np, sq, sk]
       
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/norm_factor))
        
        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
        attention_scores = attention_scores * (norm_factor)
        if not hasattr(self, 'attention_mask'):
            self.attention_mask = torch.tril(torch.ones(
    (1, seq, seq), device=attention_scores.device)).view(
            1, 1, seq, seq)
            self.attention_mask = (self.attention_mask < 0.5)
        attention_scores.masked_fill_(self.attention_mask, -10000.0)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs=attention_probs.type(input.dtype)
        attention_probs = self.attention_dropout(attention_probs)
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)
        
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        
        context_layer=context_layer.permute(1,0,2)

        output=F.linear(context_layer,self.attn_ow,self.attn_ob)
        
        res_input=input
        if self.mp_group:
            mp_rank=dist.get_rank(self.mp_group)
            world_size=dist.get_world_size(self.mp_group)
            rs_output=output.split(output.shape[1]//world_size,dim=1)[mp_rank]
            res_input=res_input.split(res_input.shape[1]//world_size,dim=1)[mp_rank]
        else:
            rs_output=output
        
        return output,res_input,rs_output
        
class GshardGate(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.top_k = args.k
        self.capacity_factor = args.f
        self.wg = torch.nn.Linear(args.M, dist.get_world_size(), bias=False)
        self.config=args
        # self.wg.weight.comm = "ep_group"
    def top1gating(self,
                logits: Tensor,
                min_capacity: int=1,
                use_tutel: bool = True):
        capacity_factor=self.capacity_factor
   
        # everything is in fp32 in this function
        gates = F.softmax(logits, dim=1)
        gates=gates.type(logits.dtype)
        capacity = _capacity(gates,
                            torch.tensor(capacity_factor),
                            torch.tensor(min_capacity))
        capacity=self.top_k*capacity
        # if (capacity%4)>0:
        #     capacity+=(4-capacity%4) 
        # Create a mask for 1st's expert per token
        # noisy gating
        indices1_s = torch.argmax( gates,
            dim=1)
        num_experts = int(gates.shape[1])
        mask1 = F.one_hot(indices1_s, num_classes=num_experts)

        mask1_rand = mask1

        assert logits.shape[0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

        top_idx = _top_idx(mask1_rand, capacity)

        new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
        mask1 = new_mask1

        if use_tutel:
            # Tutel doesn't support index values masked with zero
            # so we need to replace masked indices with -1
            indices_mask = mask1.sum(dim=1) * num_experts - 1
            indices1_s = torch.min(indices1_s, indices_mask).type(logits.dtype)

        # Compute locations in capacity buffer
        if use_tutel:
            locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
        else:
            locations1 = torch.cumsum(mask1, dim=0) - 1

        if use_tutel:
            gates1_s = (gates * mask1).sum(dim=1).type(logits.dtype)
            locations1_s = torch.sum(locations1 * mask1, dim=1).type(logits.dtype)

            return  capacity, num_experts, [indices1_s,], [locations1_s,], [gates1_s,]

        # Store the capacity location for each token
        locations1_s = torch.sum(locations1 * mask1, dim=1)

        # Normalize gate probabilities
        mask1_float = mask1.float()
        gates = gates * mask1_float

        locations1_sc = _one_hot_to_float(locations1_s, capacity)
        combine_weights = einsum("se,sc->sec", gates, locations1_sc)

        dispatch_mask = combine_weights.bool()

        return  combine_weights, dispatch_mask
    def forward(self, inp):
        
        
        inp=inp.view(-1,inp.shape[-1])
        if self.config.M>2048 and args.fp16:
            logits=F.linear(inp,self.wg.weight.half())
        else:
            logits = self.wg(inp)
        return self.top1gating(logits)
        probs = F.softmax(logits, dim=1)
        num_tokens = logits.shape[0]
        num_experts = logits.shape[1]
        capacity = int(
            self.top_k * math.ceil(num_tokens / num_experts) * self.capacity_factor
        )

        indices1 = torch.argmax(probs, dim=1)
        mask1 = F.one_hot(indices1, num_experts)
        # gate_logits_except1 = logits.masked_fill(mask1.bool(), float("-inf"))
        # indices2_s = torch.argmax(gate_logits_except1, dim=1)
        # mask2 = F.one_hot(indices2_s, num_experts)
    
        # gates1_s = (probs * mask1).sum(dim=1)
        # gates2_s = (probs * mask2).sum(dim=1)
        # norms = gates1_s + gates2_s
        # Avoid divide-by-zero
        # norms = torch.clamp(norms, min=torch.finfo(norms.dtype).eps)

        # gates1_s = gates1_s / norms
        # # gates2_s = gates2_s / norms
        # # sampled = (2 * gates2_s) > torch.rand_like(gates2_s)
        # # mask2 = mask2 * sampled.repeat(num_experts, 1).transpose(1, 0)

        locations1 = torch.cumsum(mask1, dim=0) - 1
        # # locations2 = (
        # #     torch.cumsum(mask2, dim=0) - 1 + torch.sum(mask1, dim=0, keepdim=True)
        # # )
        mask1 = mask1 * torch.lt(locations1, capacity)
        # # mask2 = mask2 * torch.lt(locations2, capacity)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        # # locations2_s = torch.sum(locations2 * mask2, dim=1)
        # # einsum("s,se->se")
        # gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)
        # print(gates1.shape)
        # # einsum("s,se->se")
        # # gates2 = gates2_s.unsqueeze(-1) * mask2.to(gates2_s.dtype)
        if self.config.M>2048 and args.fp16:
            locations1_sc = F.one_hot(locations1_s, capacity).half()
            gates=mask1.half()*probs
            combine1_sec = einsum("se,sc->sec", gates, locations1_sc)
        else:
            locations1_sc = F.one_hot(locations1_s, capacity).float()
            gates=mask1.float()*probs
            combine1_sec = einsum("se,sc->sec", gates, locations1_sc)
        # if self.config.M and args.fp16:
        #     locations1_sc = F.one_hot(locations1_s, capacity).half()
        #     combine1_sec = torch.bmm(
        #         # einsum("se,sc->sec")
        #         torch.ones((num_tokens, num_experts, 1),device=locations1_sc.device).half(),
        #         locations1_sc.unsqueeze(1),
        #     )
        # else:
        #     locations1_sc = F.one_hot(locations1_s, capacity).float()
        #     combine1_sec = torch.bmm(
        #         # einsum("se,sc->sec")
        #         torch.ones((num_tokens, num_experts, 1),device=locations1_sc.device),
        #         locations1_sc.unsqueeze(1),
        #     )
        # # combine2_sec = torch.bmm(
        # #     # einsum("se,sc->sec")
        # #     gates2.unsqueeze(-1),
        # #     locations2_sc.to(gates2.dtype).unsqueeze(1),
        # # )
        # # combine_weights = combine1_sec + combine2_sec
        combine_weights = combine1_sec
        dispatch_mask = combine_weights.bool()
        return combine_weights, dispatch_mask


class Expert(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.num_local_expert = 1
        hidden_hidden_size = args.H
        self.hidden_size = args.M
       
        self.expand_weight = nn.Parameter(
                0.001*torch.rand(self.num_local_expert, self.hidden_size, hidden_hidden_size).cuda(),requires_grad=True
            )
        if args.ffn_type==1:
            self.expand_weight1 = nn.Parameter(
                    0.001*torch.rand(self.num_local_expert, self.hidden_size, hidden_hidden_size).cuda(),requires_grad=True
                )
        self.reduce_weight = nn.Parameter(
                0.001*torch.rand(self.num_local_expert, hidden_hidden_size, self.hidden_size).cuda(),requires_grad=True
            )
        
        self.activation_fn = nn.GELU() if args.ffn_type==0 else nn.SiLU()

    def forward(self, inp):
   
        out = torch.bmm(inp, self.expand_weight)
        out = self.activation_fn(out)
        if args.ffn_tpye==1:
            out1=torch.bmm(inp, self.expand_weight1)
            out=out*out1
        out = torch.bmm(out, self.reduce_weight)
        out = out
        return out


class MixtureOfExpert(nn.Module):
    r"""
    Make the FMoETransformerMLP layer that distributes experts across
    communication group `group` to replace the original MLP layer in Megatron.
    """

    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.M
        self.gate = GshardGate(args)
        self.experts = Expert(args)

    def forward(self, inp):
        # print(inp.shape)
        inp = moe_utils._gather(_groups["es_group"], inp, 0)
        input_shape = list(inp.shape)
        inp = inp.reshape(-1, input_shape[-1])
        # inp = moe_utils.scatter(_groups["mp_group"], inp, 0)
        combine_weights, dispatch_mask = self.gate(inp)
        dispatch_mask = dispatch_mask.to(inp.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
        E, C, S = dispatch_mask.size()
        # einsum("sec,sm->ecm")
        dispatched_input = torch.matmul(dispatch_mask, inp)  # -> E,C,M
        # print(dispatched_input.shape)
        # dispatched_input = dispatched_input.contiguous()
        # dispatched_output = torch.zeros_like(dispatched_input)
        # dist.all_to_all_single(
        #     dispatched_output, dispatched_input, group=_groups["ep_group"]
        # )
        # dist.all_to_all_single(
        #     dispatched_output, dispatched_input, None, None, _groups["ep_group"]
        # )
        # dispatched_input = torch.tile(
        #     dispatched_input, (1, _groups["es_group"].size(), 1)
        # )
        # dispatched_input = moe_utils.alltoall(
        #     torch.distributed.group.WORLD, dispatched_input
        # )
        dispatched_input = moe_utils.alltoall(_groups["ep_group"], dispatched_input)
        dispatched_input = self.experts(dispatched_input)
        # if _groups["emp_group"] is not None:
        #     # dispatched_input = moe_utils.reduce(
        #     #     _groups['emp_group'], dispatched_input)
        #     # dispatched_input = moe_utils.scatter(
        #     #     _groups['emp_group'], dispatched_input, 1)
        #     dispatched_input = moe_utils.reduce_scatter(
        #         _groups["emp_group"], dispatched_input, 1
        #     )
        dispatched_input = moe_utils.alltoall(_groups["ep_group"], dispatched_input)
        # dispatched_input = dispatched_input.view(E, _groups["es_group"].size(), C, -1)
        # dispatched_input = dispatched_input.sum(dim=1)
        inp = combine_weights.view(S, E * C).mm(
            dispatched_input.view(E * C, self.hidden_size)
        )
        # # print(dispatched_input.view(E, C, self.hidden_size)[:, :5, 0])
        # inp = moe_utils.gather(_groups["mp_group"], inp, 0)
        inp = inp.reshape(input_shape)
        inp = moe_utils._split(_groups["es_group"], inp, 0)
        return inp
class MoeTransformer(nn.Module):

    def __init__(self, args,mp_group=None,orders_unit=None):
        super().__init__()
        self.mp_group=mp_group
       
        self.hidden_size = args.M
    def measure(self,x):
        
       
        global e1
        global e2
        output=torch.empty_like(x)
        ag_input=1.0*x
    
        o1=torch.empty_like(x)
        tensor_list=torch.empty((dist.get_world_size(group=self.mp_group) * ag_input.shape[0],ag_input.shape[1],)+ag_input.shape[2:], dtype=ag_input.dtype, device=ag_input.device)
        
        torch.distributed.barrier()
       
        e1.record()

        dist.all_to_all_single(output, x, None, None, _groups["ep_group"])
      
        e2.record()
       
        torch.cuda.synchronize()

        t1=(e1.elapsed_time(e2))
        torch.distributed.barrier()
        torch.cuda.synchronize()
        e1.record()
        global s
    
        with torch.cuda.stream(s):
            if self.mp_group is not None:
                self.mp_group._allgather_base(tensor_list,ag_input).wait()
    
        dist.all_to_all_single(o1, output, None, None, _groups["ep_group"])
        torch.cuda.current_stream().wait_stream(s)
       
        e2.record()
        torch.cuda.synchronize()
        t2=(e1.elapsed_time(e2))
        return t1,t2

    def simplea2a(self,dispatched_input,combine_weights,E,C,S,output,gates_=None):
        # output = torch.empty_like(dispatched_input)
        dist.all_to_all_single(output, dispatched_input, None, None, _groups["ep_group"])
        return output,combine_weights,E,C,S
   
    def expt(self,dispatched_input,combine_weights,E,C,S):
        exp_out=self.experts(dispatched_input)
        output = torch.empty_like(exp_out)
        return exp_out,combine_weights,E,C,S,output
    def forward(self,inp):
        # return self.forward_with_order(inp)
        if args.d1>1 or args.orderT==1:
            return self.forward_with_order(inp)


        ####test
        # inp_list=inp.split(inp.shape[0]//2,dim=0)
        # inp0=inp_list[0]
        
        # inp1=inp_list[1]

        # attention_out0,res_inp0,rs_out0=self.attention(inp0)
        # attention_out0=moe_utils.event_record(attention_out0,events_list[0][0])

        # with torch.cuda.stream(s):
        #     attention_out0=moe_utils.event_wait(attention_out0,events_list[0][0])
        #     rs_out0,res_inp0=self.simple_mp_operate(attention_out0,res_inp0,rs_out0)
        #     rs_out0=moe_utils.event_record(rs_out0,events_list[1][0])
        # rs_out0=moe_utils.event_wait(rs_out0,events_list[1][0])
        # dispatched_input0,combine_weights,E,C,S=self.gate_op(rs_out0,res_inp0)
        # with torch.cuda.stream(s):
        #     dispatched_input0=moe_utils.custom_compress(dispatched_input0.contiguous())
        #     dispatched_input0 = moe_utils.custom_a2a(dispatched_input0)
        
        

        # attention_out1,res_inp1,rs_out1=self.attention(inp1)
        # attention_out1=moe_utils.event_record(attention_out1,events_list[0][1])
        
        
        # with torch.cuda.stream(s1):
        #     dispatched_input0=moe_utils.custom_decompress(dispatched_input0)
        #     with torch.cuda.stream(s):
                
        #         dispatched_input0=moe_utils.event_record(dispatched_input0,events_list[3][0])
        #     attention_out1=moe_utils.event_wait(attention_out1,events_list[3][0])
        #     attention_out1=moe_utils.event_wait(attention_out1,events_list[0][1])
        #     rs_out1,res_inp1=self.simple_mp_operate(attention_out1,res_inp1,rs_out1)
        
        # dispatched_input0,combine_weights,E,C,S=self.expt(dispatched_input0,combine_weights,E,C,S)
        # dispatched_input0=moe_utils.event_record(dispatched_input0,events_list[4][0])
        # with torch.cuda.stream(s1):
        #     dispatched_input0=moe_utils.event_wait(dispatched_input0,events_list[4][0])
        #     dispatched_input0=moe_utils.custom_compress(dispatched_input0.contiguous())
        #     dispatched_input0 = moe_utils.custom_a2a(dispatched_input0)
        # dispatched_input0=moe_utils.custom_decompress(dispatched_input0)
        # dispatched_input0,tensor_list=self.before_gather(dispatched_input0,combine_weights,E,C,S)
        # with torch.cuda.stream(s1):
        #     dispatched_input0=self.simple_gather_op(dispatched_input0,tensor_list)

        # torch.cuda.current_stream().wait_stream(s)
        # torch.cuda.current_stream().wait_stream(s1)
        # return dispatched_input0.mean()+rs_out1.mean()
        ####test


    
        attention_out,res_inp,rs_out=self.attention(inp)

        rs_out,res_inp=self.mp_operate(attention_out,res_inp,rs_out)

        dispatched_input,E,C,S=self.simple_gate_op(rs_out,res_inp)
        def expert_fn(expert_input):
            return self.experts(expert_input).reshape(-1,expert_input.shape[-2],expert_input.shape[-1]).contiguous()
                 
        dispatched_input=a2a_ffn_overlap_new(dispatched_input,expert_fn=expert_fn, a2a_ffn_overlap_degree=args.d2,  group=_groups["ep_group"])

        dispatched_input=self.gather_op(dispatched_input,E,C,S)


        return dispatched_input
class MoE1(nn.Module):
    r"""
    Make the FMoETransformerMLP layer that distributes experts across
    communication group `group` to replace the original MLP layer in Megatron.
    """

    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.M
        self.gate = GshardGate(args)
        self.experts = Expert(args)

    def forward(self, inp):
        input_shape = list(inp.shape)
        inp = inp.reshape(-1, input_shape[-1])
        # inp = moe_utils.scatter(_groups["mp_group"], inp, 0)
        combine_weights, dispatch_mask = self.gate(inp)
        dispatch_mask = dispatch_mask.to(inp.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
        E, C, S = dispatch_mask.size()
        # einsum("sec,sm->ecm")
        dispatched_input = torch.matmul(dispatch_mask, inp)  # -> E,C,M
        # print(dispatched_input.shape)
        dispatched_input = torch.tile(
            dispatched_input, (1, _groups["es_group"].size(), 1)
        )
        dispatched_input = moe_utils.alltoall(
            torch.distributed.group.WORLD, dispatched_input
        )
        dispatched_input = self.experts(dispatched_input)
        dispatched_input = moe_utils.alltoall(_groups["ep_group"], dispatched_input)
        dispatched_input = dispatched_input.view(E, _groups["es_group"].size(), C, -1)
        dispatched_input = dispatched_input.sum(dim=1)
        inp = combine_weights.view(S, E * C).mm(
            dispatched_input.view(E * C, self.hidden_size)
        )
        inp = inp.reshape(input_shape)
        return inp

def decorate_trace_handler(args, rank):
    def trace_handler(prof):
        # print(prof.key_averages().table(
        #     sort_by="self_cuda_time_total", row_limit=-1))
        if rank == 0:
            # print(prof.events())
            prof.export_chrome_trace(
                "test{}.json".format(rank))
    return trace_handler
import os, time
def mp_init(args):
    mp_group=None
    if args.mp_size==1:
        return None
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    num_tp_group=dist.get_world_size() // args.mp_size
    for i in range(num_tp_group):
        size=args.mp_size
        tp_cnt=i*size
        ranks=list(range(tp_cnt, tp_cnt + size))
        _tp_group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            mp_group = _tp_group
        
    return mp_group
def rerange_bak_orders(orders):

    d1=0
    d2=0
    for item in orders:
        if item==0:
            d1+=1
        if item==2:
            d2+=1
    out_order=[]
    comp=[]
    comm=[]
    gate_pre=[0]*d1
    rs_size_store=0
    rs_count=0
    base=d1*d2
    rs_thr=base//d1
    for i in range(d2):
        rs_size_store+=base//d2
        while rs_size_store>=rs_thr:
            rs_size_store-=rs_thr
            gate_pre[rs_count]=i
            rs_count+=1
    for item in orders:
        if item in[0,2,4,6]:
            comp.append(item)
        else:
            comm.append(item)
    counts=[0]*8
    comp_heads=0
    comm_heads=0
    len_comp=len(comp)
    len_comm=len(comm)
    debug=0
    if dist.get_rank()==0:
        print(orders)
    while True:
        debug+=1
        # if dist.get_rank()==0 and debug<30:
        #     print(out_order)
        if len(out_order)==len(orders):
            break
        for i_m in range(comm_heads,len_comm+1):
            if i_m==len_comm:
                break
            item=comm[i_m]
          
            num=counts[item]
            if item==1:
                
                if counts[item+1]>gate_pre[num]:
                    out_order.append(item)
                    counts[item]+=1
                else:
                    break
            elif counts[item+1]>num:
                out_order.append(item)
                counts[item]+=1
            else:
                break
        comm_heads=i_m
        if comp_heads<len_comp:
            item2=comp[comp_heads]
            n2=counts[item2]
           
            if item2==6 or counts[item2+1]>n2:
                out_order.append(item2)
                counts[item2]+=1
                comp_heads+=1   
    return out_order 
def rerange_orders(orders):
    d1=0
    d2=0
    for item in orders:
        if item==0:
            d1+=1
        if item==2:
            d2+=1
    out_order=[]
    comp=[]
    comm=[]
    gate_pre=[0]*d2
    rs_size_store=0
    rs_count=0
    base=d1*d2
    rs_thr=base//d2
    for i in range(d1):
        rs_size_store+=base//d1
        while rs_size_store>=rs_thr:
            rs_size_store-=rs_thr
            gate_pre[rs_count]=i
            rs_count+=1
    for item in orders:
        if item in[0,2,4,6]:
            comp.append(item)
        else:
            comm.append(item)
    counts=[0]*8
    comp_heads=0
    comm_heads=0
    len_comp=len(comp)
    len_comm=len(comm)
    while True:
        if len(out_order)==len(orders):
            break
        for i_m in range(comm_heads,len_comm+1):
            if i_m==len_comm:
                break
            item=comm[i_m]
            if item<10:
                num=counts[item]
                if counts[item-1]>num:
                    out_order.append(item)
                    counts[item]+=1
                else:
                    break
            else:
                i1=item//10
                i2=item%10
                n1=counts[i1]
                n2=counts[i2]
                if counts[i1-1]>n1 and counts[i2-1]>n2:
                    out_order.append(item)
                    counts[i1]+=1
                    counts[i2]+=1
                else:
                    break
        comm_heads=i_m
        if comp_heads<len_comp:
            item2=comp[comp_heads]
            n2=counts[item2]
            if item2==2:
                if counts[item2-1]>gate_pre[n2]:
                    out_order.append(item2)
                    counts[item2]+=1
                    comp_heads+=1
            elif item2==0 or counts[item2-1]>n2:
                out_order.append(item2)
                counts[item2]+=1
                comp_heads+=1
    
    return out_order
def out_log(args,outtimes,forwardtimes,str_time_sets):
    if dist.get_rank() == 0:
        global output_file
        print(outtimes)
        # with open('test.log', 'a+') as f:
        with open(output_file, "a+") as f:
            f.write(
                str(outtimes)
                + ","
                +str(forwardtimes)
                + ","
                + str(args.B)
                + ","
                + str(args.L)
                + ","
                + str(args.M)
                + ","
                + str(args.H)
                + ","
                + str(args.k)
                + ","
                + str(args.f)
                + ","
                + str(args.heads)
                + ","
                + str(args.mp_size)
                + ","
                + str(args.d1)
                + ","
                + str(args.d2)
                + ','
                +str(args.orderT)
                +','
                + str(str_time_sets)
                + "\n"
            )
from numpy import mean
def trim_mean(data, trim_percent):
    """Compute the trimmed mean of a list of numbers.

    Args:
        data (list): List of numbers.
        trim_percent (float): Percentage of data to trim.

    Returns:
        float: Trimmed mean.
    """
    assert trim_percent >= 0.0 and trim_percent <= 1.0
    n = len(data)
    # Account for edge case of empty list
    if len(data) == 0:
        return 0
    data.sort()
    k = int(round(n * (trim_percent)))
    return mean(data[k:n - k])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--B", type=int)
    parser.add_argument("--L", type=int)
    parser.add_argument("--M", type=int)
    parser.add_argument("--H", type=int)
    parser.add_argument("--ffn_type", type=int,default=0)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--es_size", type=int, default=1)
    parser.add_argument("--mp_size", type=int, choices=[1,2,4,8])
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--f", type=float)
    parser.add_argument("--d", type=str, default=None)
    parser.add_argument("--d1", type=int, default=2)
    parser.add_argument("--d2", type=int, default=4)
    parser.add_argument("--v", type=int, choices=[0,1,2])
    parser.add_argument("--orderT", type=int, choices=[0,1])
    dct = {0: MoeTransformer, 1: MoeTransformer,2:MoeTransformer}
    nae = {0: "moeT", 1: "moeT",2:"moeT"}
    args = parser.parse_args()
    
    # if args.mp_size==1:
    #     args.fp16=True
    # else:
    #     args.fp16=False
    args.fp16=False
    if args.d is not None and args.d!='-1':
        args.d1=int(args.d[0])
        args.d2=int(args.d[1])
    ####read orders
    str_time_sets=""
   
    
    global orders
    global order_bak
    op_orders=[]
    random_order=[]
    opb_orders=[]
    d1_set=[8,8,8,8,4,4,4,4,2,2,2,2,1,1,1,1]
    d2_set=[8,4,2,1,8,4,2,1,8,4,2,1,8,4,2,1]
    dcounts=0
    ocounts=0
    m_t=0
  
        
   
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    dist_rank = dist.get_rank()
    torch.distributed.barrier()
    global output_file
    output_file = nae[args.v] + "_" + str(dist.get_world_size()) + ".log"

    
    torch.manual_seed(772002 + dist_rank)
    moe_init(args)

    
    
    global s
    s = torch.cuda.Stream(priority=0)

    e1=torch.cuda.Event(enable_timing=True)
    e2=torch.cuda.Event(enable_timing=True)
    e3=torch.cuda.Event(enable_timing=True)
    e4=torch.cuda.Event(enable_timing=True)
    mp_group=mp_init(args)
    model = dct[args.v](args,mp_group).cuda()
    times = []
    
    
    # output_file = "debug.log"
    
    
   
    
    shape1=args.B//args.d2*args.L//args.mp_size*args.M

    

   
    ####read time_table###
    flag=False
    with open("time_table.txt")as f:
        lines=f.readlines()
    for line in lines:
        tmp=line.strip('\n').split(',')
        s2,m2=int(tmp[0]),int(tmp[1])
        if shape1==s2 and args.mp_size==m2:
            real_t1=float(tmp[2])
            real_t2=float(tmp[3])
            flag=True
    ####measure
    
    if flag==False:
        real_t1=[]
        real_t2=[]
        x = torch.randn([args.B//args.d2*args.L//args.mp_size, args.M], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            x=x.detach()
            model.eval()
            for i in range(50):
                t1,t2=model.measure(x)
            
                real_t1.append(t1)
                real_t2.append(t2)
        real_t1=np.array(real_t1)
        real_t2=np.array(real_t2)
        real_t1=trim_mean(real_t1,0.2)
        real_t2=trim_mean(real_t2,0.2)
        if dist.get_rank()==0:
            with open("time_table.txt", "a+") as f:
                f.write(str(x.shape[0]*x.shape[1])+','+str(args.mp_size)+','+str(real_t1)+','+str(real_t2)+'\n')

    # time_sets[3]=real_t1
    # time_sets[5]=real_t1
    # time_sets[8]=real_t2
    
    # str_time_sets="["
    # for item in time_sets:
    #     str_time_sets=str_time_sets+str(item) +" "
    # out_log(args,0,0,str_time_sets)
    
    # exit(0)
    