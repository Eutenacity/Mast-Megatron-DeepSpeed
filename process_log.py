import numpy as np

with open('08configured_results.log','r') as f:
    data=f.readlines()

small_count=0
mode_value1=['88','84','82','81','48','44','42','41','28','24','22','21','18','14','12','11']
mode_value=[]
for item in mode_value1:
    mode_value.append(item+'0')
    mode_value.append(item+'1')
verse_dict={}
for idx,item in enumerate(mode_value):
    verse_dict[item]=idx
len_mode=len(mode_value)
set_nums=0
line_count=0
t=[0 for _ in range(len_mode)]
t2=[0 for _ in range(len_mode)]
times=[0 for _ in range(len_mode)]
fastcount=[0 for _ in range(len_mode)]
per_degree_compare=[0 for _ in range(len_mode)]
per_degree_compare_max=[-1 for _ in range(len_mode)]

x_set1=[8,8,8,8,4,4,4,4,2,2,2,2,1,1,1,1]
x_set2=[8,4,2,1,8,4,2,1,8,4,2,1,8,4,2,1]
def inc_rate(v1,v2):
    return (v1-v2)/v1
maxid=-1
max_v=-1
whole_a_set=[]
d_change=[0,1,2,2,4,4,4,8,8]


d1_bonus=[]
d2_bonus=[]
d1_bonus_bound=[]
dsame_bonus=[]
ddif_bonus=[]
delta=[]
counts=0

times_real=[]
relate=[0,1,2,4,5,6]
hist=[0,0,0,0,0]

aaa=[]
bbb=[]
improve0=[]
improve=[]
improve1=[]
improve2=[]

#table 4
imp_tutel_mp=[]
imp_pipemoe_mp=[]

imp_tutel_nomp=[]
imp_pipemoe_nomp=[]

#table7
imp_mastpipemoe_np=[]
imp_mast_np=[]

#table9
imp_same_diff_mp1=[]
imp_same_diff_mp2=[]
imp_same_diff_mp4=[]
imp_same_diff_mp8=[]
total_counts=0
max_v=0
for line in data:
    a=line.split(',')
    str_mode=a[-4]+a[-3]+a[-2].strip('\n')
    if a[-2]!='0':
        times_tmp=float(a[0])
        times_tmp=times_tmp if times_tmp>1 else times_real[-1]
        aaa.append(times_tmp)
        bbb.append([])
    if str_mode == mode_value[small_count]:  
        times_real.append(float(a[0]))
        small_count+=1
        if small_count==len_mode:
            oxminv=9999#pipemoe
            oominv=9999#no pipeline
            ttminv=9999#tutel
            d1minv=9999#mast+pipemoe
            orminv=9999#d is same(mast)
            ordminv=9999#d is different
            npminv=9999#mast w/o order (naive pipe)
            best_id=-1
            for idx,item in enumerate(mode_value):
                
                time_tmp=times_real[idx] if times_real[idx]>1 else times_real[idx-1]
                if item[0]=='1' and item[1]=='1' and item[2]=='0':
                    oominv=min(oominv,time_tmp)
                if item[0]=='1' and item[1]=='2' and item[2]=='0':
                    ttminv=min(ttminv,time_tmp)
                if item[0]=='1' and item[2]=='0':
                    if oxminv>time_tmp:
                        oxminv=time_tmp
                        best_id=item[1]
                
                if  item[2]=='0' and item[0]==item[1]:
                    npminv=min(npminv,time_tmp)
                # if  item[2]=='1':
                if item[0]==item[1] and item[2]=='1':
                    orminv=min(orminv,time_tmp)
                if  item[2]=='1':
                    ordminv=min(ordminv,time_tmp)
            for idx,item in enumerate(mode_value):
                time_tmp=times_real[idx] if times_real[idx]>1 else times_real[idx-1]
                if item[0]==best_id and item[1]==best_id:
                    d1minv=min(d1minv,time_tmp)
            op=oxminv/orminv
            ot=ttminv/orminv

            dp=oxminv/d1minv
            dt=ttminv/d1minv

            dd=(orminv-ordminv)/orminv



            
            if a[9]in ['1']:
                npminv=oxminv
                imp_tutel_nomp.append(ttminv/orminv)
                imp_pipemoe_nomp.append(oxminv/orminv)
                imp_same_diff_mp1.append(inc_rate(orminv,ordminv)*100)
            if a[9] in ['8','4','2']:
                imp_tutel_mp.append(ttminv/orminv)
                imp_pipemoe_mp.append(oxminv/orminv)
                
                imp_mastpipemoe_np.append(npminv/d1minv)
                imp_mast_np.append(npminv/orminv)
            

            
            if a[9] == '2':
                imp_same_diff_mp2.append(inc_rate(orminv,ordminv)*100)


            
            if a[9] == '4':
                imp_same_diff_mp4.append(inc_rate(orminv,ordminv)*100)
            if a[9] == '8':
                imp_same_diff_mp8.append(inc_rate(orminv,ordminv)*100)
                if max_v<imp_same_diff_mp8[-1]:
                    max_v = imp_same_diff_mp8[-1]
                    # print(orminv)
                    # print(ordminv)
                    # print(line_count)
            times_real=[]
        small_count%=len_mode
        
    else:
        times_real=[]
        print(line_count)
        print(str_mode)
        small_count=0
        print("log_format_error, maybe the process break down during the experiment. Please restart the experiment.")
        exit(0)
    line_count+=1
print("Note: Please ensure the experiments are all conducted with the same number of GPUs")
print("Table 4 Averaged speedups of Mast over Tutel and PipeMoE")

print("mast over tutel with mp:"+str(np.array(imp_tutel_mp).mean()))
print("mast over pipemoe with mp:"+str(np.array(imp_pipemoe_mp).mean()))

print("mast over tutel w/o mp:"+str(np.array(imp_tutel_nomp).mean()))
print("mast over pipemoe w/o mp:"+str(np.array(imp_pipemoe_nomp).mean()))
print("")

print("Table 7 averaged speedups of our Mast and Mast+PipeMoE over Mast w/o optimized orders")

print("mast+pipemoe over mast w/o order:"+str(np.array(imp_mastpipemoe_np).mean()))
print("mast over mast w/o order:"+str(np.array(imp_mast_np).mean()))
print("")

print("Table 9 Performance improvement of separately searching the attention pipeline degree for the attention and Reduce-Scatter operations as well as the MoE pipeline degree for the MoE layer on average (%)")

print("diff over same mp1 mean:"+str(np.array(imp_same_diff_mp1).mean())+", max:"+str(np.array(imp_same_diff_mp1).max()))
print("diff over same mp2 mean:"+str(np.array(imp_same_diff_mp2).mean())+", max:"+str(np.array(imp_same_diff_mp2).max()))
print("diff over same mp4 mean:"+str(np.array(imp_same_diff_mp4).mean())+", max:"+str(np.array(imp_same_diff_mp4).max()))
print("diff over same mp8 mean:"+str(np.array(imp_same_diff_mp8).mean())+", max:"+str(np.array(imp_same_diff_mp8).max()))
