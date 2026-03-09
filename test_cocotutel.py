import argparse
import torch
from torch import nn
from torch import distributed as dist
from torch.nn import functional as F
from torch._C._distributed_c10d import ProcessGroup

import math
import numpy as np
_groups = None
from torch import Tensor
from tutel import moe as tutel_moe

import mm_ar
class MMARStatue:
    initialized = False

    @staticmethod
    def init(group: dist.ProcessGroup) -> None:
        # Initialize NCCL
        if not MMARStatue.initialized:
            mp_rank = dist.get_rank(group)
            nccl_unique_id_size = mm_ar.get_nccl_unique_id_size()
            nccl_unique_id = torch.zeros([nccl_unique_id_size], dtype=torch.int8).cpu()
            mp_size = dist.get_world_size(group)
            world_rank = dist.get_rank()
            if mp_rank == 0:
                mm_ar.get_nccl_unique_id(nccl_unique_id)
            nccl_unique_id = nccl_unique_id.cuda()
            dist.broadcast(nccl_unique_id,world_rank//mp_size * mp_size, group)
            mm_ar.init_nccl(
                nccl_unique_id.cpu(), mp_size, mp_rank
            )
            MMARStatue.initialized = True
class _A2A_FFN_BakByHand(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  x,bak_fun,input):
        
        ctx.bak_fun=bak_fun
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return None,None,ctx.bak_fun(grad_output)
def bak_a2a_ffn_ol(grad_output,a2afn,a2a_ffn_overlap_degree,experts_input,experts_output):
    split_dim=1
    split_size = grad_output.shape[1] // a2a_ffn_overlap_degree
    grad_split = list(grad_output.split(split_size, dim=1))
    for i in range(a2a_ffn_overlap_degree):
        grad_split[i] = grad_split[i].contiguous()
    s.wait_stream(torch.cuda.current_stream())
    for i in range(a2a_ffn_overlap_degree):
        with torch.cuda.stream(s):
            grad_split[i] = a2afn(grad_split[i])
            events_list[3][i].record()
    for i in range(a2a_ffn_overlap_degree):
        events_list[3][i].wait()
        experts_input[i].retain_grad()
        torch.autograd.backward(experts_output[i], grad_tensors=grad_split[i])
        grad_split[i]=experts_input[i].grad
        events_list[4][i].record()
        with torch.cuda.stream(s):
            events_list[4][i].wait()
            grad_split[i] = a2afn(grad_split[i])
    torch.cuda.current_stream().wait_stream(s)
    
    output = torch.cat(grad_split, dim=1)

    return output
def a2a_ffn_overlap_new(input,expert_fn,a2a_ffn_overlap_degree,a2afn):
    split_dim=1
    
    split_size = input.shape[1] // a2a_ffn_overlap_degree
    input_split = list(input.split(split_size, dim=1))
    
    experts_input = []
    experts_output = []
    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = input_split[i].contiguous()
    s.wait_stream(torch.cuda.current_stream())
    for i in range(a2a_ffn_overlap_degree):
        with torch.cuda.stream(s):
            input_split[i] = a2afn(input_split[i])
            events_list[3][i].record()
    for i in range(a2a_ffn_overlap_degree):
        events_list[3][i].wait()
        input_split[i] = input_split[i].detach()
        input_split[i].requires_grad= True
        experts_input.append(input_split[i])
       
        input_split[i]=expert_fn(input_split[i])
        
        experts_output.append(input_split[i])
        events_list[4][i].record()
        with torch.cuda.stream(s):
            events_list[4][i].wait()
            input_split[i] = a2afn(input_split[i])
    
    torch.cuda.current_stream().wait_stream(s)
    
    output = torch.cat(input_split, dim=1)

    output = output.detach()
    output.requires_grad = True
    def bak_fun(grad):
        return bak_a2a_ffn_ol(grad,a2afn,a2a_ffn_overlap_degree,experts_input,experts_output)
    output = _A2A_FFN_BakByHand.apply(output,bak_fun,input)
    return output

def _split(group: ProcessGroup, input_, dim):
    world_size = dist.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    assert input_.size(dim) % world_size == 0, '{} is not divisible by {}'.format(
        input_.size(dim), world_size)
    dim_size = input_.size(dim) // world_size
    input_list = torch.split(input_, dim_size, dim=dim)
    rank = dist.get_rank(group)
    output = input_list[rank].contiguous()
    return output
def _ori_gather(group: ProcessGroup, input_, dim, tensor_list_tmp=None):
    world_size = dist.get_world_size(group) if group is not None else 1
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    ag_input=input_
    if tensor_list_tmp is not None:
        tensor_list=tensor_list_tmp
    else:
        tensor_list=torch.empty((ag_input.shape[0],dist.get_world_size(group=group) * ag_input.shape[1],)+ag_input.shape[2:], dtype=ag_input.dtype, device=ag_input.device)

    group._allgather_base(tensor_list,ag_input).wait()
    output=tensor_list
    return output
def _ori_reduce_scatter(group: ProcessGroup, input_, rs_output, dim):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    assert input_.size(dim) % world_size == 0, '{} is not divisible by {}'.format(
        input_.size(dim), world_size)
    torch.distributed._reduce_scatter_base(rs_output, input_, group=group)
    return rs_output
class _OriReduceScatter(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, group, input_, rs_output,dim):
        return _ori_reduce_scatter(group, input_, rs_output,dim)

    @staticmethod
    def forward(ctx, group, input_, rs_output,dim):
        ctx.dim = dim
        ctx.group = group
        ctx.tensor_list=torch.empty_like(input_)

        return _ori_reduce_scatter(group, input_,rs_output, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return None, _ori_gather(ctx.group, grad_output, ctx.dim,ctx.tensor_list), None,None
def ori_reduce_scatter(group, input_, rs_output,dim):
    return _OriReduceScatter.apply(group, input_, rs_output,dim)
class _OriGatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, group,  input_, dim,tensor_list_tmp):
        return _ori_gather(group, input_, dim,tensor_list_tmp)

    @staticmethod
    def forward(ctx, group,  input_, dim,tensor_list_tmp):
        ctx.dim = dim
        ctx.group = group
        ctx.grad_dummy=input_
        return _ori_gather(group, input_, dim,tensor_list_tmp)

    @staticmethod
    def backward(ctx, grad_output):
        return None,ctx.grad_dummy, None,grad_output
def ori_gather(group, input_, dim,tensor_list_tmp):
    return _OriGatherFromModelParallelRegion.apply(group, input_, dim,tensor_list_tmp)
class CudaEventTimer(object):
    def __init__(self, start_event: torch.cuda.Event, end_event: torch.cuda.Event):
        self.start_event = start_event
        self.end_event = end_event

    def get_elapsed_msec(self):
        torch.cuda.current_stream().wait_event(self.end_event)
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)


class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""
    class Timer:
        """Timer."""
        def __init__(self, name):
            self.name_ = name
            self.started_ = False
            self.event_timers = []
            self.start_event = None
            self.elapsed_records = None

        def start(self):
            """Start the timer."""
            assert not self.started_, f"{self.name_} timer has already been started"
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            self.started_ = True

        def stop(self, reset=False, record=False):
            """Stop the timer."""
            assert self.started_, "timer is not started"
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self.event_timers.append(CudaEventTimer(self.start_event, end_event))
            self.start_event = None
            self.started_ = False

        def _get_elapsed_msec(self):
            self.elapsed_records = [et.get_elapsed_msec() for et in self.event_timers]
            self.event_timers.clear()
            return sum(self.elapsed_records)

        def reset(self):
            """Reset timer."""
            self.started_ = False
            self.start_event = None
            self.elapsed_records = None
            self.event_timers.clear()

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self._get_elapsed_msec()
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

        def mean(self,trim_percent=0.1):
            self.elapsed(reset=False)
            m,s=trim_mean(self.elapsed_records, trim_percent)
            return m,s

    def __init__(self):
        self.timers = {}

    def get_timers(self):
        return self.timers

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    @staticmethod
    def memory_usage():
        alloc = "mem_allocated: {:.4f} GB".format(torch.cuda.memory_allocated() /
                                                  (1024 * 1024 * 1024))
        max_alloc = "max_mem_allocated: {:.4f} GB".format(
            torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024))
        cache = "cache_allocated: {:.4f} GB".format(torch.cuda.memory_cached() /
                                                    (1024 * 1024 * 1024))
        max_cache = "max_cache_allocated: {:.4f} GB".format(
            torch.cuda.max_memory_cached() / (1024 * 1024 * 1024))
        return " | {} | {} | {} | {}".format(alloc, max_alloc, cache, max_cache)



    def get_mean(self, names, normalizer=1.0, reset=True):
        """Get the mean of a group of timers."""
        assert normalizer > 0.0
        means = {}
        for name in names:
            if name in self.timers:
                elapsed_time = (self.timers[name].mean() * 1000.0 / normalizer)
                means[name] = elapsed_time
        return means
import numpy as np
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
        return 0,0
    data.sort()
    k = int(round(n * (trim_percent)))
    m=np.mean(data[k:n - k])
    s=np.std(data[k:n - k])
    return m,s

class OrdersFunc:
    def __init__(self,s,s1,d1,d2,mp_group,batch_size,events_list) -> None:
        self.s=s
        self.s1=s1
        self.d1=d1
        self.d2=d2
        self.mp_group=mp_group
        self.mp_size=dist.get_world_size(group=mp_group) if mp_group is not None else 1
        self.B=batch_size
        self.events_list=events_list
    def backward_with_order(self,grad):
        bak_func_list=self.bak_func_list
        order_bak=self.order_bak
        s=self.s
        d1=self.d1
        inp=self.input
        input_tensors=self.input_tensors
        output_tensors=self.output_tensors
        events_list=self.events_list
        bak_counts=[0]*8
        
        bak_gate_pre=[0]*self.d1
        rs_size_store=0
        rs_count=0
        base=self.d1*self.d2
        rs_thr=base//self.d1
        for i in range(self.d2):
            rs_size_store+=base//self.d2
            while rs_size_store>=rs_thr:
                rs_size_store-=rs_thr
                bak_gate_pre[rs_count]=i
                rs_count+=1

        gard_arg_af=[]
        gard_arg=list(grad.split(grad.shape[0]//self.d2,dim=0))
        def get_bak_num(id):
            out=bak_counts[id]
            bak_counts[id]+=1
            return out
        def check_bak_event(id,num):
            if id==6:
                return
            events_list[id+1][num].wait()
        def bakprogress(item,inarg):
            num=get_bak_num(item)
            if item==1:#rs_bak
                check_bak_event(item,bak_gate_pre[num])
                
            else:
                check_bak_event(item,num)
          
            
            inarg[num]=bak_func_list[item](input_tensors[item][num],output_tensors[item][num],inarg[num],item)
            input_tensors[item][num]=None
            output_tensors[item][num]=None
            events_list[item][num].record()
           
            return num    
        micro_b=self.B//d1
        rs_store=torch.tensor([],dtype=inp.dtype).cuda()
        res_store=torch.tensor([],dtype=inp.dtype).cuda()
        self.pre_ptr=0
        rs_thr=inp.shape[0]*inp.shape[1]//d1//self.mp_size

        post_ptr=0
        for i in range(d1):
            gard_arg_af.append((None,))
        # bak_func_list=[self.backward_step,self.mp_bak,self.backward_step,self.a2a_bak,self.backward_step,self.a2a_bak,self.before_gather_bak,None]

        for item in order_bak:
  
            inarg=gard_arg_af if item in [0,1] else gard_arg
            num=bak_counts[item]


            if item in [0,2,4]:
                if item ==0:
                    num=bakprogress(item,inarg)
                elif item ==2:
                    num=bakprogress(item,inarg)
                else:
                    num=bakprogress(item,inarg)
                if item ==2:
                    while self.pre_ptr<=num:
                    
                        rs_store=torch.cat([rs_store,gard_arg[self.pre_ptr][0].reshape(-1,gard_arg[self.pre_ptr][0].shape[-1])],dim=0)
                        res_store=torch.cat([res_store,gard_arg[self.pre_ptr][1].reshape(-1,gard_arg[self.pre_ptr][1].shape[-1])],dim=0)
                        self.pre_ptr+=1
                    while rs_store.shape[0]>=rs_thr:
                        slice_rs=rs_store[:rs_thr]
                        rs_store=rs_store[rs_thr:]
                        slice_res=res_store[:rs_thr]
                        res_store=res_store[rs_thr:]

                        slice_rs=slice_rs.reshape(micro_b,-1,slice_rs.shape[-1])
                        if self.mp_group is not None:

                            tensor_list=torch.empty((micro_b, dist.get_world_size(group=self.mp_group)* slice_rs.shape[1],)+slice_rs.shape[2:], dtype=slice_rs.dtype, device=slice_rs.device)
                        else:
                            tensor_list=None
                        gard_arg_af[post_ptr]=(slice_rs,slice_res.reshape(micro_b,-1,slice_res.shape[-1]),tensor_list)
                        post_ptr+=1
                    events_list[item][num].record()

            elif item in [3,5]:
                with torch.cuda.stream(s):
                    
                    bakprogress(item,inarg)
                   
            elif item == 1:
                with torch.cuda.stream(s):
                    bakprogress(item,inarg)
            elif item in [6]:
                bakprogress(item,inarg)
        torch.cuda.current_stream().wait_stream(s)
        for i in range(self.d1):
            gard_arg_af[i]=gard_arg_af[i][0]
        grad_output=torch.cat(gard_arg_af)   
        self.input.backward(grad_output)
    def forward_with_order(self,inp,func_list,orders,bak_func_list,order_bak):
        self.bak_func_list=bak_func_list
        self.order_bak=order_bak
        self.input=inp
        inp=inp.detach()
        input_tensors=[[[] for _ in range(self.d1)] for _ in range(2)]+[[[]for _ in range(self.d2)] for _ in range(6)]
        output_tensors=[[[] for _ in range(self.d1)]for _ in range(2)]+[[[]for _ in range(self.d2)]for _ in range(6)]
        d1=self.d1
        d2=self.d2
        events_list=self.events_list
        s=self.s
        s1=self.s1
        rs_thr=inp.shape[0]*inp.shape[1]//d2//self.mp_size
        # func_list=[self.attention,self.simple_mp_operate,self.gate_op,self.simplea2a,self.expt,self.simplea2a,self.before_gather,self.simple_gather_op]
        inp_list=inp.split(inp.shape[0]//d1,dim=0)
      
        
        counts=[0,0,0,0,0,0,0,0]
        outarg=[]
        outarg_after=[]
     
        

        ###准备输入参数
        for i in range(d1):
            inp_list[i].requires_grad=True
            outarg.append((inp_list[i],))
        for i in range(d2):
            outarg_after.append((None,))
        self.rs_store=torch.tensor([],dtype=inp.dtype).cuda()
        self.res_store=torch.tensor([],dtype=inp.dtype).cuda()
        self.pre_ptr=0
        def check_event(id,num):
            if id==0:
                return 
            events_list[id-1][num].wait()
        def get_num(id):
            out=counts[id]
            counts[id]+=1
            return out
        def progress(item,inarg):
            
            num=get_num(item)
            
            if item==2:#gate
                check_event(item,gate_pre[num])
                while self.pre_ptr<=gate_pre[num]:
                    self.rs_store=torch.cat([self.rs_store,outarg[self.pre_ptr][0].reshape(-1,outarg[self.pre_ptr][0].shape[-1])],dim=0)
                    self.res_store=torch.cat([self.res_store,outarg[self.pre_ptr][1].reshape(-1,outarg[self.pre_ptr][1].shape[-1])],dim=0)
                    self.pre_ptr+=1
                slice_rs=self.rs_store[:rs_thr]
                self.rs_store=self.rs_store[rs_thr:]
                slice_res=self.res_store[:rs_thr]
                self.res_store=self.res_store[rs_thr:]

                slice_rs=slice_rs.detach()
                slice_rs.requires_grad=True

                slice_res=slice_res.detach()
                slice_res.requires_grad=True

                inarg[num]=(slice_rs,slice_res)
            else:
                check_event(item,num)
            if item in [0,2,4,6]:
                # inarg[num][0].retain_grad()
                input_tensors[item][num]=inarg[num]
                
                # if dist.get_rank()==0:
                #     print(str(item)+'************'+str(input_tensors))
            #     print(str(item)+str(inarg[num][0].shape))
            inarg[num]=func_list[item](*inarg[num])
            if item in [1,3,5]:
                if item in [1]:
                    tmp=inarg[num][0].detach()
                    tmp1=inarg[num][1].detach()
                    tmp.requires_grad=True
                    tmp1.requires_grad=True
                    inarg[num]=(tmp,tmp1)+inarg[num][2:]
                else:
                    tmp=inarg[num][0].detach()
                    tmp.requires_grad=True
                    inarg[num]=(tmp,)+inarg[num][1:]
                
            if item == 7:
                inarg[num]=inarg[num].detach()
                inarg[num].requires_grad=True
            events_list[item][num].record()
            if item in [0,2,4,6]:
                output_tensors[item][num]=inarg[num]
            return num
        
        rs_count=0
        gate_pre=[0]*d2
        rs_size_store=0
        prenumag=0
        preitemag=-1
        prenuma2a=0
        preitema2a=-1
       
        for item in orders:
            # if dist.get_rank()==0:
            #     print("begin:{}".format(item))
            ###attention,reduce
            if item in [0]:
                progress(item,outarg)
            elif item in [1]:
                
                with torch.cuda.stream(s):
                    num=progress(item,outarg)
                # print(outarg[num][0].shape)
                rs_size_store+=outarg[num][0].shape[0]*outarg[num][0].shape[1]
                while rs_size_store>=rs_thr:
                    rs_size_store-=rs_thr
                    gate_pre[rs_count]=num
                    rs_count+=1
            ###gate,a2a,exp,gather
            elif item in [2,4,6]:
                progress(item,outarg_after)
            elif item in [3,5,7]:
                
                with torch.cuda.stream(s):
                    if preitema2a!=-1:
                        events_list[preitema2a][prenuma2a].wait()
                    if preitemag!=-1:
                        events_list[preitemag][prenumag].wait()
                    num=progress(item,outarg_after)
                if item in [3,5]:
                    prenuma2a=num
                    preitema2a=item
            elif item in [37,73,57,75]:
                # progress(5,outarg_after)
                # progress(6,outarg_after)
                i1=item//10
                i2=item%10
                # if dist.get_rank()==0:
                #     print("**")
                #     print(counts[7])
                with torch.cuda.stream(s1):
                    if preitema2a!=-1:
                        events_list[preitema2a][prenuma2a].wait()
                with torch.cuda.stream(s):
                    if preitemag!=-1:
                        events_list[preitemag][prenumag].wait()
                if i2==7:
                    with torch.cuda.stream(s):
                        num1=progress(i1,outarg_after)
                    with torch.cuda.stream(s1):
                        num2=progress(i2,outarg_after)
                    prenuma2a=num1
                    preitema2a=i1
                    prenumag=num2
                    preitemag=i2
                else:
                    with torch.cuda.stream(s):
                        num2=progress(i2,outarg_after)
                    with torch.cuda.stream(s1):
                        num1=progress(i1,outarg_after)
                    prenuma2a=num2
                    preitema2a=i2
                    prenumag=num1
                    preitemag=i1

        
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.current_stream().wait_stream(s1)
        output=torch.cat(outarg_after)
        output=output.detach()
        output.requires_grad=True
        output=_BakByHand.apply(output,self)
        self.input_tensors=input_tensors
        self.output_tensors=output_tensors
        return output
        
class _BakByHand(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  x,obj):
        
        ctx.obj=obj
        return x

    @staticmethod
    def backward(ctx, grad_output):
        ctx.obj.backward_with_order(grad_output)
        return None,None

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
class _mmar_bak(torch.autograd.Function):
    @staticmethod
    def forward(ctx, context_layer,weight,bias):

        output = torch.empty( args.B,  args.L,  args.M,dtype= torch.float16,device = context_layer.device)
        mm_ar.mm_ol_ar( args.B,  args.L,  args.M,  weight, bias, context_layer.half() ,  output)

        ctx.weight=weight
        ctx.context_layer = context_layer
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_shape = grad_output.shape
        return torch.mm(grad_output.view(-1,grad_shape[-1]),ctx.weight.float()).view(ctx.context_layer.shape),torch.mm(grad_output.view(-1,grad_shape[-1]).permute(1,0).contiguous(),ctx.context_layer.view(-1,ctx.context_layer.shape[-1])).half(),None

class Attention(nn.Module):
    def __init__(self, args,mp_group) -> None:
        super().__init__()

        self.layer_norm=torch.nn.LayerNorm(args.M,eps=1e-5,elementwise_affine=True)
        self.hidden_size=args.M
        qkv_size_per_partition = (args.M// args.mp_size) * 3
        out_size_per_partition = args.M // args.mp_size
        
       
        self.attn_qkvw = nn.Parameter(0.001*torch.randn(qkv_size_per_partition,args.M).cuda(),
                                    requires_grad=True)
    
        self.attn_qkvb = nn.Parameter(torch.zeros(qkv_size_per_partition).cuda(),
                                    requires_grad=True)
        
        self.attn_ow = nn.Parameter(0.001*torch.randn(args.M,
                                                out_size_per_partition,dtype=torch.float16 ).cuda(),
                                    requires_grad=True)

        self.attn_ob = nn.Parameter(torch.zeros(args.M,dtype=torch.float16 ).cuda(),
                                    requires_grad=True)
        
        self.num_attention_heads_per_partition = args.heads // args.mp_size
        self.hidden_size_per_partition = args.M // args.mp_size
        self.hidden_size_per_attention_head = args.M // args.heads

        self.mp_group = mp_group
        self.norm_factor=math.sqrt(args.M// args.heads)
        self.attention_dropout = torch.nn.Dropout(0.2)
    def forward(self,input):
        
        seq=input.shape[1]
        
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
        attention_probs=attention_probs.type(query_layer.dtype)
        # attention_probs = self.attention_dropout(attention_probs)
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
        
        
        context_layer=context_layer.permute(1,0,2).contiguous()
        # output=F.linear(context_layer,self.attn_ow,self.attn_ob)

        output=_mmar_bak.apply(context_layer,self.attn_ow,self.attn_ob)
        

        return output
        
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
        
        logits = self.wg(inp)
        return self.top1gating(logits)


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
        self.ffn_type=args.ffn_type
        self.activation_fn = nn.GELU() if args.ffn_type==0 else nn.SiLU()

    def forward(self, inp):

        inp=inp.view(self.num_local_expert, -1, self.hidden_size)
        out = torch.bmm(inp, self.expand_weight)
        out = self.activation_fn(out)
        if self.ffn_type==1:
            out1=torch.bmm(inp, self.expand_weight1)
            out=out*out1
        out = torch.bmm(out, self.reduce_weight)
        out = out
        return out

class _split_bak_ag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, context_layer,mp_group,simple_gather_op):

        ctx.mp_group = mp_group
        ctx.simple_gather_op=simple_gather_op
        return _split(mp_group,context_layer,dim=1)

    @staticmethod
    def backward(ctx, grad_output):

        return ctx.simple_gather_op(grad_output),None,None

class MoeTransformer(nn.Module):

    def __init__(self, args,mp_group=None,orders_unit=None):
        super().__init__()
        self.attention=Attention(args,mp_group)
        self.mp_group=mp_group
        self.gate = GshardGate(args)
        self.experts = Expert(args)
        self.hidden_size = args.M
        self.layer_norm=torch.nn.LayerNorm(args.M,eps=1e-5,elementwise_affine=True)
        self.orders_unit=orders_unit

    def mp_operate(self,attention_output,res_input,rs_output):
        
        if self.mp_group is not None:
            mp_group = self.mp_group
            list_size = dist.get_world_size(group=mp_group)
            assert (
                (attention_output.shape[0]*attention_output.shape[1]) % list_size == 0
                and (attention_output.shape[0]*attention_output.shape[1]) >= list_size
            )
            rs_output=ori_reduce_scatter(mp_group,attention_output,rs_output,dim=1)
           
        return rs_output,res_input        
    def simple_mp_operate(self,attention_output,res_input,rs_output):
        
        if self.mp_group is not None:
            mp_group = self.mp_group
            list_size = dist.get_world_size(group=mp_group)
            assert (
                (attention_output.shape[0]*attention_output.shape[1]) % list_size == 0
                and (attention_output.shape[0]*attention_output.shape[1]) >= list_size
            )
            torch.distributed._reduce_scatter_base(rs_output, attention_output, group=mp_group)
        else:
            rs_output=attention_output
        return rs_output,res_input
    def simple_gate_op(self,attention_output):
      
        self.attention_out_shape=list(attention_output.shape)
        
        attention_output=self.layer_norm(attention_output)
        
        attention_output=attention_output.view(-1,attention_output.shape[-1])

        C, E, indices_, locations_, gates_=self.gate(attention_output)
   
        S, M = attention_output.size(0), attention_output.size(1)
        if not hasattr(self, '_tutel_dispatcher'):
            self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C,M,dispatch_dtype=attention_output.dtype)

            self._tutel_dispatcher.update(indices_, locations_,gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(attention_output)
            
        else:
            self._tutel_dispatcher.update(indices_, locations_,gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(attention_output)
         
        
        return dispatched_input,E,C,M
    def gate_op(self,rs_output,res_input):
        
        attention_output=rs_output+res_input
       
        
        self.attention_out_shape=list(attention_output.shape)
        
        attention_output=self.layer_norm(attention_output)
        
        attention_output=attention_output.view(-1,attention_output.shape[-1])

        #combine_weights, dispatch_mask=self.gate(attention_output)
        # dispatch_mask = dispatch_mask.to(attention_output.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
        # E, C, S = dispatch_mask.size()
        # # einsum("sec,sm->ecm")
        # dispatched_input = torch.matmul(dispatch_mask, attention_output)
        
        C, E, indices_, locations_, gates_=self.gate(attention_output)
        combine_weights=gates_[0]
        combine_weights=combine_weights.detach()
        combine_weights.requires_grad=True
        S, M = attention_output.size(0), attention_output.size(1)
        if not hasattr(self, '_tutel_dispatcher'):
            self._tutel_dispatcher = [tutel_moe.fast_dispatcher(E, C,M,dispatch_dtype=attention_output.dtype) for _ in range(args.d2)]

            self._tutel_dispatcher[self.gate_counts].update(indices_, locations_, [combine_weights], capacity=C)
            dispatched_input = self._tutel_dispatcher[self.gate_counts].encode(attention_output)
            
        else:
            self._tutel_dispatcher[self.gate_counts].update(indices_, locations_, [combine_weights], capacity=C)
            dispatched_input = self._tutel_dispatcher[self.gate_counts].encode(attention_output)
         
        self.gate_counts+=1
        output=torch.empty_like(dispatched_input)
        
        return dispatched_input,combine_weights,E,C,M,output,gates_[0]
    def gather_op(self,dispatched_input,E,C,M):
        output = self._tutel_dispatcher.decode(dispatched_input.view(E * C, M))
        output=output.reshape(self.attention_out_shape)
        tensor_list=torch.empty((output.shape[0],dist.get_world_size(group=self.mp_group) * output.shape[1],)+output.shape[2:], dtype=output.dtype, device=output.device)
       
        if self.mp_group is not None:
            output=ori_gather(self.mp_group,output,1,tensor_list)

        return output
    def simple_gather_op(self,output,tensor_list=None):
        if self.mp_group is not None:
            ag_input=output
            if tensor_list is None:
                tensor_list=torch.empty((ag_input.shape[0],dist.get_world_size(group=self.mp_group) * ag_input.shape[1],)+ag_input.shape[2:], dtype=ag_input.dtype, device=ag_input.device)
            self.mp_group._allgather_base(tensor_list,ag_input).wait()
        else:
            tensor_list=output
        output=tensor_list
        
        return output

    def before_gather(self,dispatched_input,combine_weights,E,C,M):
        # output = combine_weights.view(S, E * C).mm(
        #     dispatched_input.view(E * C, self.hidden_size)
        # )
        # output=output.reshape(self.attention_out_shape)
        output = self._tutel_dispatcher[self.comb_counts].decode(dispatched_input.view(E * C, M))
        output=output.reshape(self.attention_out_shape)
        if self.mp_group is not None:
            tensor_list=torch.empty((output.shape[0],dist.get_world_size(group=self.mp_group) * output.shape[1],)+output.shape[2:], dtype=output.dtype, device=output.device)
        else:
            tensor_list=None
        self.comb_counts+=1
        return (output,tensor_list)
    def before_gather_bak(self,input_tensor, output_tensor, output_tensor_grad,item=6):
        if self.mp_group is not None:
            out=_split(self.mp_group,output_tensor_grad,dim=1)
        else:
            out=output_tensor_grad
        grad_outpus=self.backward_step(input_tensor, output_tensor, [out,output_tensor_grad],6)
        
        return  *grad_outpus,torch.empty_like(grad_outpus[0])
        
    def backward_step(self,input_tensor, output_tensor, output_tensor_grad,item=0):

        if isinstance(input_tensor,tuple):
            input_tensor=list(input_tensor)
        if isinstance(output_tensor,tuple):
            output_tensor=list(output_tensor)
        if isinstance(output_tensor_grad,tuple):
            output_tensor_grad=list(output_tensor_grad)
        # Retain the grad on the input_tensor.
        unwrap_input_tensor_grad = False
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
            unwrap_input_tensor_grad = True
        for x in input_tensor:
            if x is not None and torch.is_tensor(x) and x.requires_grad:
                x.retain_grad()

        if not isinstance(output_tensor, list):
            output_tensor = [output_tensor]
        if not isinstance(output_tensor_grad, list):
            output_tensor_grad = [output_tensor_grad]

        if item in [0]:
            torch.autograd.backward(output_tensor[0:2], grad_tensors=output_tensor_grad[0:2])
        elif item in [2]:
            torch.autograd.backward([output_tensor[0],output_tensor[-1]], grad_tensors=output_tensor_grad[0:2])
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])
   

        # Collect the grad of the input_tensor.
        input_tensor_grad = [None]
        if input_tensor is not None:
            input_tensor_grad = []
            for x in input_tensor:
                if x is None or torch.is_tensor(x)==False or (x.requires_grad==False):
                    input_tensor_grad.append(None)
                else:
                    input_tensor_grad.append(x.grad)
        if item == 4:
            input_tensor_grad[1]=output_tensor_grad[1]
        if unwrap_input_tensor_grad:
            input_tensor_grad = input_tensor_grad[0]
        else:
            input_tensor_grad= tuple(input_tensor_grad)
       
        # if debug:
        #     if dist.get_rank()==0:
        #         print(input_tensor_grad[0].size())
        if item in [4]:
            return *input_tensor_grad,torch.empty_like(input_tensor_grad[0])
        return input_tensor_grad

    def forward_with_order(self,inp):
        self.gate_counts=0
        self.comb_counts=0
        global orders
        global order_bak
        func_list=[self.attention,self.simple_mp_operate,self.gate_op,self.simplea2a,self.expt,self.simplea2a,self.before_gather,self.simple_gather_op]
        bak_func_list=[self.backward_step,self.mp_bak,self.backward_step,self.a2a_bak,self.backward_step,self.a2a_bak,self.before_gather_bak,None]

        return self.orders_unit.forward_with_order(inp,func_list,orders,bak_func_list,order_bak)
    def a2a_bak(self,input_tensor, output_tensor, output_tensor_grad,item=0):
        return self.simplea2a(*output_tensor_grad)
    def mp_bak(self,input_tensor, output_tensor, output_tensor_grad,item=0):
        rs_output_g,res_input_g,tensor_list=output_tensor_grad
        if self.mp_group is not None:
            out=self.simple_gather_op(rs_output_g,tensor_list)
        else:
            out=rs_output_g
        return out,res_input_g,None
    def measure(self,x,flag=False):
        
        x1=(x,)
        global e1
        global e2
        global e3
        func_list=[self.attention,self.simple_mp_operate,self.gate_op,self.simplea2a,self.expt,self.simplea2a,self.before_gather,self.simple_gather_op]
        times=np.array([0.0]*9)
        # tmp_input=torch.rand(1).cuda()
        # tmp_output=torch.rand(1).cuda()
        for idx,func in enumerate(func_list):
            self.gate_counts=0
            self.comb_counts=0
            # torch.cuda.synchronize()
            # dist.all_to_all_single(tmp_output, tmp_input, None, None, _groups['ep_group'])
            torch.distributed.barrier()
            torch.cuda.synchronize()
            
            if idx==5:
                x1tmp=x1
            if idx==7:
                x2tmp=x1
            e1.record()
            
            x1=func(*x1)
            # torch.distributed.barrier()
            e2.record()
            torch.cuda.synchronize()
            t1=e1.elapsed_time(e2)
            times[idx]=t1
            if flag and idx==1:
                return times

        torch.distributed.barrier()
        torch.cuda.synchronize()
        e1.record()
        global s
        self.gate_counts=0
        self.comb_counts=0
        # if dist.get_rank()==0:
        #     print(x2tmp[0].shape)
        #     print(x1tmp[0].shape)
        with torch.cuda.stream(s):
            self.simple_gather_op(*x2tmp)
        self.simplea2a(*x1tmp)
        torch.cuda.current_stream().wait_stream(s)
        # torch.distributed.barrier()
        e2.record()
        torch.cuda.synchronize()
        times[8]=(e1.elapsed_time(e2))
        return times

    def simplea2a(self,dispatched_input,combine_weights,E,C,S,output,gates_=None):
        # output = torch.empty_like(dispatched_input)
        # torch.distributed.barrier()
        dist.all_to_all_single(output, dispatched_input, None, None, _groups["ep_group"])
        return output,combine_weights,E,C,S
   
    def expt(self,dispatched_input,combine_weights,E,C,S):
        exp_out=self.experts(dispatched_input)
        output = torch.empty_like(exp_out)
        return exp_out,combine_weights,E,C,S,output
    def forward(self,inp):
        # return self.forward_with_order(inp)
        # if args.d1>1 or args.orderT==1 or (args.d1==1 and args.d2==1):
        #     return self.forward_with_order(inp)


      
        ####normal overlap(pipemoe/tutel)


       
        attention_out=self.attention(inp)+inp
        list_size = dist.get_world_size(group=mp_group)
        def gather_op(inputs):
            return self.simple_gather_op(inputs,None)
        attention_out=_split_bak_ag.apply(attention_out,mp_group,gather_op)
        # attention_out = attention_out.split( attention_out.shape[1]//list_size, dim=1)[dist.get_rank(mp_group)]

        dispatched_input,E,C,S=self.simple_gate_op(attention_out)
        def expert_fn(expert_input):
            return self.experts(expert_input).reshape(-1,expert_input.shape[-2],expert_input.shape[-1]).contiguous()
        def a2a_fn(inputs):
            output = torch.empty_like(inputs)
            dist.all_to_all_single(output, inputs, None, None, _groups["ep_group"])
            return output
        dispatched_input = a2a_ffn_overlap_new(dispatched_input,expert_fn=expert_fn, a2a_ffn_overlap_degree=args.d2, a2afn=a2a_fn)

        dispatched_input=self.gather_op(dispatched_input,E,C,S)


        return dispatched_input

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
        # print(outtimes)
        print("time cost of each component, copy and paste to opt_order.py line 727")
        print(str_time_sets.strip('[').strip(' ').replace(' ',','))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--measure_time", action="store_true")
    parser.add_argument("--B", type=int)
    parser.add_argument("--L", type=int)
    parser.add_argument("--M", type=int)
    parser.add_argument("--H", type=int)
    parser.add_argument("--ffn_type", type=int,default=0) #0: 2ffn, 1: llama type
    parser.add_argument("--att_type", type=int,default=0) #0: nromal, 2: flash
    parser.add_argument("--heads", type=int)
    parser.add_argument("--es_size", type=int, default=1)
    parser.add_argument("--mp_size", type=int, choices=[1,2,4,8])
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--f", type=float)
    parser.add_argument("--d", type=str, default=None)
    parser.add_argument("--d1", type=int, default=2)
    parser.add_argument("--d2", type=int, default=4)
    parser.add_argument("--v", type=int, default=0,choices=[0,1,2])
    parser.add_argument("--orderT", type=int, choices=[0,1]) # choose the default order (0) or optimized order (1)
    dct = {0: MoeTransformer, 1: MoeTransformer,2:MoeTransformer}
    nae = {0: "test", 1: "imp1",2:"moeT"}
    args = parser.parse_args()
    
  
    if args.d is not None and args.d!='-1':
        args.d1=int(args.d[0])
        args.d2=int(args.d[1])
    ####read orders
    str_time_sets=""
    order_path="orders.txt"
    if args.mp_size==1:
        order_path="orders_mp1.txt"
    if os.path.exists(order_path):
        with open(order_path)as f:
            lines=f.readlines()
    else:
        lines=[]
    global orders
    global order_bak
    op_orders=[]
    opb_orders=[]
    d1_set=[8,8,8,8,4,4,4,4,2,2,2,2,1,1,1,1]
    d2_set=[8,4,2,1,8,4,2,1,8,4,2,1,8,4,2,1]
    dcounts=0
    ocounts=0
    m_t=0
    for line in lines:
        ocounts+=1
        tmp=line.split('|')
        seq,m1,h1,heads1,mp1=int(tmp[1]),int(tmp[2]),int(tmp[3]),int(tmp[6]),int(tmp[7])
        odr=tmp[9].strip('\n').split(',')
        if ocounts%2==0:
       
            if (args.L==seq)and  m1==args.M and h1==args.H and heads1==args.heads and mp1==args.mp_size:
                if args.d1==d1_set[dcounts] and args.d2==d2_set[dcounts]:
                    op_orders=np.array(podr).astype(np.int64)
                    opb_orders=np.array(odr).astype(np.int64)
                    m_t=pm_t
                    break
                else:
                    dcounts+=1
        pseq,pm1,ph1,pheads1,pmp1,podr=seq,m1,h1,heads1,mp1,odr
        pm_t=float(tmp[8])
        
    input_d=args.orderT
    d1=0
    d2=0
    if input_d==1:
        
        for item in op_orders:
            if item==0:
                d1+=1
            if item ==2:
                d2+=1
        args.d1=d1
        args.d2=d2
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    dist_rank = dist.get_rank()
    global output_file
    if args.measure_time:
        if args.mp_size==1:
            output_file ='measure_time_mp1'+".log"
        else:
            output_file ='measure_time'+".log"
    else:
        # output_file = nae[args.v] + "_" + str(dist.get_world_size()) + ".log"
        if dist.get_world_size() == 8 :
            output_file = "8configured_results.log"
        else:
            output_file = "configured_results.log"
   
    torch.manual_seed(772002 + dist_rank)
    moe_init(args)

    if dist.get_rank()==0:
        print(op_orders)
   
    if dist.get_rank()==0:
        print("rerange end")

    global s
    global s1
    s1 = torch.cuda.Stream(priority=0)
    s = torch.cuda.Stream(priority=0)
    global events_list
  
    events_list=[[torch.cuda.Event(enable_timing=True) for _ in range(args.d1)]for _ in range(2)]+[[torch.cuda.Event(enable_timing=True) for _ in range(args.d2)]for _ in range(6)]
    e1=torch.cuda.Event(enable_timing=True)
    e2=torch.cuda.Event(enable_timing=True)
    e3=torch.cuda.Event(enable_timing=True)
    e4=torch.cuda.Event(enable_timing=True)
    mp_group=mp_init(args)
    x = torch.randn([args.B, args.L, args.M], dtype=torch.float32, device="cuda",requires_grad=True)
    MMARStatue.init(mp_group)
    model = dct[args.v](args,mp_group,OrdersFunc(s,s1,args.d1,args.d2,mp_group,args.B,events_list)).cuda()
    times = []
    
    
    # output_file = "debug.log"
    
    
    prof = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=False,
    with_modules=True,
    schedule=torch.profiler.schedule(
        wait=10,
        warmup=10,
        active=2),
    on_trace_ready=decorate_trace_handler(args, dist.get_rank())
    ) 
    
    ####measure
    if args.measure_time:
        xshape=x.shape
        real_t1=[]
        real_t2=[]
        
        xsplit1=x[:xshape[0]//args.d1]
        xsplit2=x[:xshape[0]//args.d2]
        
        with torch.no_grad():
            x1=xsplit1.detach()
            xsplit1=xsplit2.detach()
            model.eval()
            for i in range(50):
                t1=model.measure(x1,flag=True)
                t2=model.measure(xsplit1)
            
                real_t1.append(t1)
                real_t2.append(t2)
        real_t1=np.array(real_t1)
        real_t2=np.array(real_t2)
        time_sets=[0]*9
        time_sets[0]=trim_mean(real_t1[:,0],0.15)[0]
        time_sets[1]=trim_mean(real_t1[:,1],0.15)[0]
        for i in range(2,9):
            time_sets[i]=trim_mean(real_t2[:,i],0.15)[0]
        # real_t1=np.array(real_t1).mean(axis=0)
        # real_t2=np.array(real_t2).mean(axis=0)
        # time_sets=np.concatenate((real_t1[:2],real_t2[2:]),axis=0)
        
        # time_sets=torch.tensor(time_sets).cuda()
        # print(str(time_sets[1])+','+str(dist.get_rank()))
        # torch.distributed.all_reduce(time_sets)
        # time_sets/=dist.get_world_size()
        # time_sets=time_sets.detach().to('cpu').numpy()
        str_time_sets="["
        for item in time_sets:
            str_time_sets=str_time_sets+str(item) +" "
        out_log(args,0,0,str_time_sets)
        exit(0)
    # ####get default order
    
    tem_orders=[0,1]*args.d1+[2,3]*args.d2+[4,5]*args.d2+[6]*args.d2+[7]*args.d2
    if args.mp_size==1 :
        if args.d1==args.d2:
            tem_orders=[0,1,2,3]*args.d1+[4,5]*args.d2+[6]*args.d2+[7]*args.d2
        elif args.d1>args.d2:
            tms=args.d1//args.d2
            tem_orders=([0,1]*tms+[2,3])*args.d2+[4,5]*args.d2+[6]*args.d2+[7]*args.d2
        else:
            tms=args.d2//args.d1
            tem_orders=([0,1]+[2,3]*tms)*args.d1+[4,5]*args.d2+[6]*args.d2+[7]*args.d2
    tem_orders=rerange_orders(tem_orders)
  
 
    order_list=[tem_orders,op_orders]

    orders=order_list[input_d]

    order_list_bak=[tem_orders,opb_orders]
    order_bak_tp=order_list_bak[input_d]
    order_bak=[]
    len_orders=len(order_bak_tp)
    for i in range(len_orders):
        if order_bak_tp[len_orders-1-i]>10:
            i1=order_bak_tp[len_orders-1-i]//10
            i2=order_bak_tp[len_orders-1-i]%10
            item_s=i1 if i2==7 else i2
            order_bak.append(item_s)
        elif order_bak_tp[len_orders-1-i]==7:
            pass
        else:
            order_bak.append(order_bak_tp[len_orders-1-i])
    if dist.get_rank()==0:
        print(tem_orders)
        print(op_orders)
        print(orders)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    outtimes=[]
    forwardtimes=[]
  
    x=x.cuda()
   
    test_timer=SynchronizedWallClockTimer()

    
    for i in range(50):
        torch.distributed.barrier()
        
        
        if dist.get_rank() == 0:
            test_timer('forward').start()
        
       
        # torch.cuda.synchronize()
        output=model(x)
        loss=output.mean()
        loss.backward() 
        
        torch.cuda.synchronize()
        if dist_rank == 0:
            test_timer('forward').stop()
            # print("step:", i)
            # prof.step()
            pass
        

    if dist_rank == 0:
        tmout=test_timer('forward').mean(0.2)
        print(tmout)
        outtimes,forwardtimes=tmout
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
                +str(input_d)
                +','
                + str(str_time_sets)
                + "\n"
            )
