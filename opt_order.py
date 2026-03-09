from typing import Any
import optuna
import numpy as np
import math
import argparse
class Best_Order:
    def __init__(self,d1,d2,time_set,back=True,layer=1,grad_time=0,gradbytes=0) -> None:
        self.layer=layer
        self.back=back
        self.d1=d1
        self.d2=d2
        assert d1==d2
        self.md=max(d1,d2)
        self.time_set=time_set
        #only support 8 operations in one layer now
        #including attention, reduce_scatter/all_reduce, 
        #gating(compuation before alltoall), alltoall, 
        #expert, alltoall, before gather and all-gather
        self.n=8
        self.f=[i for i in range(self.md)]
        self.g=[1 for i in range(self.md)]
        
        
        self.grad_time= grad_time
        self.gradbytes = gradbytes
        
        self.A2AAG_FLAG=False
        if self.time_set[8]>((self.time_set[5]+self.time_set[3])/2.+self.time_set[7]):
            self.A2AAG_FLAG=False
        ##get default order
        time_sets_1=time_set.copy()
        time_sets_1[6]=time_set[7]
        time_sets_1[7]=time_set[8]
        time_sets_1[8]=time_set[6]


        tem_orders=[0,1]*d1+[2,3]*d2+[4,5]*d2+[6]*d2+[7]*d2
        self.tem_orders=tem_orders*self.layer
        self.tem_layers=[]
        for i in range(self.layer):
            self.tem_layers+=[i]*(self.md*self.n)
        
        self.default_v,self.default_order,self.default_layer=self.get_value_w_orders(self.tem_orders,self.tem_layers,flag=True)
         
    def fa(self,x):
        if self.f[x]==x:
            return x
        f1=self.fa(self.f[x])
        # self.f[x]=f1
        # self.g[f1]+=self.g[x]
        return f1
    def optimal_object(self,trial):
        t_input=[]
        for i in range(self.layer*self.n*self.md):
            name=str(i)
            t_input.append(trial.suggest_int(name,0,self.md-1,1))
        orders,layers=self.decode_opt_input(t_input)
        # print(orders)
        return self.get_value_w_orders(orders,layers)
    def start_optuna(self,n_trails,algo):

        #定义使用TPE或者GP
        if algo == "TPE":
            algo = optuna.samplers.TPESampler(n_startup_trials = 15, n_ei_candidates = 20)
        elif algo == "GP":
            from optuna.integration import SkoptSampler
            
            algo = SkoptSampler(skopt_kwargs={'base_estimator':'GP', #选择高斯过程
                                            'n_initial_points':30, #初始观测点10个
                                            'acq_func':'EI'} #选择的采集函数为EI，期望增量
                            )
        elif algo=="CmaEs":
            algo = optuna.samplers.CmaEsSampler()
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        #实际优化过程，首先实例化优化器
        study = optuna.create_study(sampler = algo #要使用的具体算法
                                    , direction="minimize" #优化的方向，可以填写minimize或maximize
                               )
        #开始优化，n_trials为允许的最大迭代次数
        #由于参数空间已经在目标函数中定义好，因此不需要输入参数空间
        study.optimize(self.optimal_object #目标函数
                    , n_trials=n_trails #最大迭代次数（包括最初的观测值的）
                    , show_progress_bar=True #要不要展示进度条呀？
                    )

        #可直接从优化好的对象study中调用优化的结果
        #打印最佳参数与最佳损失值
        # print("\n","\n","best params: ", study.best_trial.params,
        #     "\n","\n","best score: ", study.best_trial.values,
        #     "\n")
       
        if study.best_trial.values[0]>=(self.default_v-(1e-5)):
            return self.default_order,self.default_layer,self.default_v
        else:
            o,l=self.decode_opt_input(study.best_trial.params.values())
            _,o,l=self.get_value_w_orders(o,l,flag=True)

            return o,l, study.best_trial.values
   
    def decode_opt_input(self,inputs):
        self.f=[i for i in range(self.md)]
        self.g=[1 for i in range(self.md)]
        counts=[self.n*self.layer]*self.md
        orders=[]
        layers=[]
        for item in inputs:
            tmpid=item
            if counts[tmpid]<=0:
                tmpid=self.fa(tmpid)
            v=self.n*self.layer-counts[tmpid]

            orders.append(v%self.n)
            layers.append(v//self.n)
            counts[tmpid]-=1
            if counts[tmpid]==0:
                
                fa1=self.fa(tmpid)
                tv=999
                for idx in range(self.md):
                    id_tmp=self.fa(idx)
                    val=self.g[id_tmp]
                    if id_tmp!=fa1 and counts[id_tmp]>0 and val<tv:
                        tv=val
                        tid=id_tmp
                    
                if tv<999:
                    fa2=self.fa(tid)
                    self.f[fa1]=fa2
                    self.g[fa2]+=self.g[fa1]
        return orders,layers
    def rerange_orders(self,orders):
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
            for i_m in range(comm_heads,len_comm):
                
                item=comm[i_m]
                if item<10:
                    num=counts[item]
                    if counts[item-1]>num:
                        out_order.append(item)
                        counts[item]+=1
                    else:
                        comm_heads=i_m
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
                        comm_heads=i_m
                        break
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
    def get_value_back(self,orders_input,layers_input):
        len_orders=len(orders_input)
        orders=[]
        layers=[]
        for i in range(len_orders):
            if orders_input[len_orders-1-i]>10:
                i1=orders_input[len_orders-1-i]//10
                i2=orders_input[len_orders-1-i]%10
                orders.append(i2)
                orders.append(i1)

                l1=layers_input[len_orders-1-i][0]
                l2=layers_input[len_orders-1-i][1]
                layers.append(l2)
                layers.append(l1)
                # item_s=i1 if i2==7 else i2
                # orders.append(item_s)
            # elif orders_input[len_orders-1-i]==7:
            #     pass
            else:
                orders.append(orders_input[len_orders-1-i])
                layers.append(layers_input[len_orders-1-i])
        times_set=[_ for _ in self.time_set]
        # times_set[1]*=2
        times_set[0]*=2
        times_set[2]*=2
        times_set[4]*=2
        times_set[6]*=2

        task_list=[[],[]]
        for (item,item_l) in zip(orders,layers):
            if item in [0,2,4,6]:
                task_list[0].append([item,item_l])#计算
            else:
                task_list[1].append([item,item_l])#通信
        heads=[0,0]
        bcounts=[0]*(self.n*self.layer)#begin count
        counts=[0]*(self.n*self.layer)#end count
        times=0
        cur_state=[-1]*2#compute task and communication task
        cur_layer=[-1]*2
        remained=[0]*2
        merged_order=[]
        merged_layer=[]
        
        holes = []
        pre_num = []
        connect = []
        connect_flag = False
        s_flag = False
        while True:
            if (cur_state[0]==-1)and(cur_state[1]==-1)and(heads[0]>=len(task_list[0]))and(heads[1]>=len(task_list[1])):
                break
            for j in range(2):
                i=j
                ##no task in work
                if (cur_state[i]==-1 )and(heads[i]<len(task_list[i])) :
                    
                    state,state_l=task_list[i][heads[i]]
                    
                    num=bcounts[self.n*state_l+state]
                    
                    if( state==7 and state_l==self.layer-1) or (counts[self.n*state_l+state+1]>num):
                        cur_state[i]=state
                        cur_layer[i]=state_l
                        remained[i]=times_set[state]
                        heads[i]+=1
                        bcounts[self.n*state_l+state]+=1
                        merged_order.append(state)
                        merged_layer.append(state_l)
            next_flag=False
            ##step
            for i in range(2):
                if cur_state[i]==-1:
                    if i == 1 and remained[0]>0:
                        pre_num.append(len(merged_order))
                        holes.append(remained[0])
                        connect.append(connect_flag )
                        if not s_flag:
                            connect_flag = True
                        else:
                            connect_flag = False
                    else:
                        connect_flag = False
                        
                    j=1-i
                    times+=remained[j]
                    
                    counts[self.n*cur_layer[j]+cur_state[j]]+=1
                      
                    remained[j]=0
                    cur_state[j]=-1
                    next_flag=True
                    
                    break
            if (next_flag):
                s_flag = False
                continue
            s_flag = True
            if remained[0]>remained[1]:
                i=1
            else:
                i=0
            times+=remained[i]
            remained[1-i]-=remained[i]
          
            
            counts[self.n*cur_layer[i]+cur_state[i]]+=1
            remained[i]=0
            cur_state[i]=-1
        
        
        grad_time = self.grad_time*self.layer
        grad_bytes = self.gradbytes
        
        if grad_time == 0:
            return times, merged_order,merged_layer
        t_p_b = grad_time * 1.0 / grad_bytes
        new_order = []
        new_layer = []
        i = 0
      
        new_holes = []
        new_prenum = []
        for idx,item in enumerate(holes):
            if connect[idx]:
                new_holes[-1] += item
            else:
                new_holes.append(item)
                new_prenum.append(pre_num[idx])
        # print(holes)
        # print(new_holes)
        # 将两个列表配对并排序
        paired = list(zip(new_holes, new_prenum))
       
        paired.sort(key=lambda x: x[0], reverse=True)
      
        # 解包回原来的列表
        new_holes[:], new_prenum[:] = zip(*paired)
        # print(new_holes)
        n_new_share = len(new_prenum)
        pre_new_id = 0
        for idx,item in enumerate(merged_order):
            pre_n = new_prenum[i]
            new_order.append(item)
            new_layer.append(merged_layer[idx])
            if idx == pre_n-1  and grad_time>0 :
           
                if grad_time > new_holes[i]:
                    tmp_b = (int)(new_holes[i]/self.grad_time * self.gradbytes)
                    new_order.append([self.n, tmp_b])
                    grad_bytes -= tmp_b
                    grad_time = grad_bytes * t_p_b
                    
                else:
                    new_order.append([self.n,grad_bytes])
                    grad_time = 0 
                    grad_bytes = 0
                pre_new_id = len(new_order)-1
                new_layer.append(-1)
               
                i = i+1 if i<n_new_share-1 else n_new_share-1
        if grad_time >0 :
            times += grad_time
            new_order[pre_new_id][1] += grad_bytes
        return times, new_order,new_layer
    def get_value_w_orders(self,orders,layers,flag=False):
        
        times_set=self.time_set
        

        task_list=[[],[]]
        for (item,item_l) in zip(orders,layers):
            if item in [0,2,4,6]:
                task_list[0].append([item,item_l])#计算
            else:
                task_list[1].append([item,item_l])#通信
        heads=[0,0]
        bcounts=[0]*(self.n*self.layer)#begin count
        counts=[0]*(self.n*self.layer)#end count
        times=0
        cur_state=[-1]*2#compute task and communication task
        cur_layer=[-1]*2
        remained=[0]*2
        merged_order=[]
        merged_layer=[]

        
        while True:
            if (cur_state[0]==-1)and(cur_state[1]==-1)and(heads[0]>=len(task_list[0]))and(heads[1]>=len(task_list[1])):
                break
            for j in range(2):
                i=j
                ##no task in work
                if (cur_state[i]==-1 )and(heads[i]<len(task_list[i])) :
                    
                    state,state_l=task_list[i][heads[i]]
                    
                    if (i==1)and((heads[i]+1)<len(task_list[i]))and self.A2AAG_FLAG:
                        snext,snext_l=task_list[i][heads[i]+1]
                        
                        if ((state in [3,5])and (snext ==7))or((snext in [3,5])and (state ==7)):
                            if snext==7:
                                combine_flag=(counts[self.n*state_l+state-1]>bcounts[self.n*state_l+state])and(bcounts[self.n*snext_l+snext-1]>bcounts[self.n*snext_l+snext])
                            else:
                                
                                combine_flag=(counts[self.n*state_l+state-1]>bcounts[self.n*state_l+state])and(counts[self.n*snext_l+snext-1]>bcounts[self.n*snext_l+snext])
                            if combine_flag:
                                cur_state[i]=(state,snext)
                                cur_layer[i]=(state_l,snext_l)
                                remained[i]=times_set[8]
                                heads[i]+=2
                                bcounts[self.n*state_l+state]+=1
                                bcounts[self.n*snext_l+snext]+=1
                                merged_order.append(state*10+snext)
                                merged_layer.append((state_l,snext_l))
                                continue
                    num=bcounts[self.n*state_l+state]
                    
                    if( state==0 and state_l==0) or (counts[self.n*state_l+state-1]>num):
                        cur_state[i]=state
                        cur_layer[i]=state_l
                        remained[i]=times_set[state]
                        heads[i]+=1
                        bcounts[self.n*state_l+state]+=1
                        merged_order.append(state)
                        merged_layer.append(state_l)
            next_flag=False
            ##step
            for i in range(2):
                if cur_state[i]==-1:
                    

                    j=1-i
                    times+=remained[j]
                    if  isinstance(cur_state[j],tuple):
                        for s1,s2 in zip(cur_state[j],cur_layer[j]):
                            counts[self.n*s2+s1]+=1
                    else:
                        counts[self.n*cur_layer[j]+cur_state[j]]+=1
                      
                    remained[j]=0
                    cur_state[j]=-1
                    next_flag=True
                    
                    break
            if (next_flag):
                continue
            if remained[0]>remained[1]:
                i=1
            else:
                i=0
            times+=remained[i]
            remained[1-i]-=remained[i]
          
            if  isinstance(cur_state[i],tuple):
                for s1,s2 in zip(cur_state[i],cur_layer[i]):
                    counts[self.n*s2+s1]+=1
            else:
                counts[self.n*cur_layer[i]+cur_state[i]]+=1
            remained[i]=0
            cur_state[i]=-1


        # merged_order=self.rerange_orders(merged_order)
        if self.back:
            times,merged_order,merged_layer=self.get_value_back(merged_order,merged_layer)
        if flag:
            return times,merged_order,merged_layer
        return times

import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_percent", type=float,default=0.0)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--measure_time", type=str,default="")
    
    args = parser.parse_args()

    back_flag=args.backward
    real_layers=args.layers
    degree=args.d

    if args.measure_time !="":
        t1,t2,t3,t4,t5,t6,t7,t8,t9=args.measure_time.split(',')
        time_sets_raw=[float(t1),float(t2),float(t3),float(t4),float(t5),float(t6),float(t7),float(t8),float(t9)]
    else:
        # time_sets_raw=[2.5,2.0,0.25,3.064161889693317,4.25696091932409,3.044017861871159,0.1168997642748496,1.8,4.1]#mixtral 64
        # time_sets_raw=[2.5,2.0,0.25,1.45,4.25696091932409,1.45,0.1168997642748496,1.8,4.1]#mixtral 32
        # time_sets_raw=[2.5,2.0,0.25,1.45/2,4.25696091932409,1.45/2,0.1168997642748496,1.8,4.1]#mixtral 16
        # time_sets_raw=[6.3,4.4,0.29,9.8,15.9,9.8,0.1168997642748496,3.9,12.7]#mixtral 32 4k 4
        # time_sets_raw=[2.5,2.1,0.25,3.8,8.1,3.8,0.1168997642748496,2.0,4.2]#mixtral 32  2048 4
        # time_sets_raw=[1.28,1.27,0.25,1.16,1.67,1.16,0.1168997642748496,0.9,1.8]#mixtral 32  fp16 2048 4
        # time_sets_raw=[1.4549515142160303,0.8441891722819385,0.5853157621972701,0.2969543539425906,0.5622889399528503,0.29544376801041994,0.12279153001659057,0.7139294077368343,0.9422315324054045]#gpt8
        # time_sets_raw=[1.461604703875149,0.8404451766434837,0.5587557676960441,0.8910578857449925,0.5614249425775865,0.8860442322843215,0.11208470538258553,0.7209054108928231,1.2980385983691496]#gpt16
        # time_sets_raw=[1.4551444719819462,0.83977411775028,0.5623124767752254,1.3120752818444197,0.5584847103146946,1.2878889406428617,0.11783058787969981,0.7165609405321234,1.6549891724305994]#gpt32
        # time_sets_raw=[1.4,1.3,0.4,8.7,0.9,8.7,0.03,1,50]#l=4096 qwen
        # time_sets_raw = [1.8,1,0.2,1.8,0.2,1.8,0.01,0.8,50]#mixtral 4k 32g inter intra
        time_sets_raw = [1.4551444719819462,0.83977411775028,0.5623124767752254,0.65,0.65,0.5584847103146946,0.65,0.65,0.11783058787969981,0.7165609405321234,1.6549891724305994]
        # time_sets_raw=[0.6,0.58,0.16,1.234,0.408,1.234,0.1198879999711233,0.449,1.4]#64
        # time_sets_raw=[0.7,0.58,0.35,0.715,0.383,0.715,0.1198879999711233,0.49,0.9]#32
        # time_sets_raw=[1.4487557761809404,0.839639532215455,0.5801280018161324,1.4298286963911617,0.5545477674287909,1.4196103495710037,0.1198879999711233,0.7148272938588086,1.8620188306359684]#the measured time cost of attention, reduce-scatter, gate, a2a, expert, a2a, operations berfore all-gather, all-gather, simultaneously begin a2a and all-gather.
    #B=4,L=2048 100g
    time_sets=[]
    percent=args.random_percent
    for item in time_sets_raw:
        t=random.random()
        percent = percent if t >0.5 else -1*percent
        item = item + item * percent
        time_sets.append(item)
    
    if args.layers<=3 or args.direct:
        # test_obj=Best_Order(degree,degree,time_sets,back=back_flag,layer=args.layers,grad_time=4.6,gradbytes=1331200)
        test_obj=Best_Order(degree,degree,time_sets,back=back_flag,layer=args.layers,grad_time=0,gradbytes=0)

        final_o,final_l,v=test_obj.start_optuna(5000,'CmaEs')#CmaEs,TPE 

        if args.layers==1:
            if args.backward:
                print("backward order for one layer is below, please copy and paste in the megatron/model/transformer.py.mast_start, line 1455")
                print(final_o)
            else:
                print("forward order for one layer is below, please copy and paste in the megatron/model/transformer.py.mast_start, line 1456")
                print(final_o)
        if args.layers>1:
            if args.backward:
                print("backward order is below, please copy and paste in the megatron/model/transformer.py, line 2590")
                print(final_o)
                print("")
                print("corresponding backward layer_idx is below, please copy and paste in the megatron/model/transformer.py, line 2591")
                print(final_l)
            else:
                print("forward order is below, please copy and paste in the megatron/model/transformer.py, line 2588")
                print(final_o)
                print("")
                print("corresponding forward layer_idx is below, please copy and paste in the megatron/model/transformer.py, line 2589")
                print(final_l)
        exit(0)
    