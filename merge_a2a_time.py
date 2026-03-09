import numpy as np
####read time_table###
def process(M,H,heads,mp_size,d1,d2,L,B):
    shape1=B//d2*L//mp_size*M
    with open("measure_time.log")as f:
        lines=f.readlines()
        for line in lines:
            tmp=line.split(',')
            seq,m1,h1,heads1,mp1,dd1,dd2=int(tmp[3]),int(tmp[4]),int(tmp[5]),int(tmp[8]),int(tmp[9]),int(tmp[10]),int(tmp[11])
            if seq==L and m1==M and h1==H and heads1==heads and mp1==mp_size and d1==dd1 and d2==dd2:
                time_sets=tmp[-1]
                tmp=time_sets.strip('\n')[1:-1]
                tmp=tmp.split(' ')
                time_sets=np.array(tmp).astype(float)
    with open("time_table.txt")as f:
        lines=f.readlines()
    flag=True
    for line in lines:
        tmp=line.strip('\n').split(',')
        s2,m2=int(tmp[0]),int(tmp[1])
        if shape1==s2 and mp_size==m2:
            real_t1=float(tmp[2])
            real_t2=float(tmp[3])
            flag=False
    if flag:
        assert False
    time_sets[3]=real_t1
    time_sets[5]=real_t1
    time_sets[8]=real_t2
    
    str_time_sets="["
    for item in time_sets:
        str_time_sets=str_time_sets+str(item) +" "
    output_file='moeT_8.log'
    with open(output_file, "a+") as f:
            f.write(
                str(0)
                + ","
                +str(0)
                + ","
                + str(B)
                + ","
                + str(L)
                + ","
                + str(M)
                + ","
                + str(H)
                + ","
                + str(1)
                + ","
                + str(1.0)
                + ","
                + str(heads)
                + ","
                + str(mp_size)
                + ","
                + str(d1)
                + ","
                + str(d2)
                + ','
                +str(0)
                +','
                + str(str_time_sets)
                + "\n"
            )
for B in [8]:
    for mp_size in [2, 4, 8]:
        for L in [2048,1024]:
            for M in [4096, 2048 ,1024]:
                for H in [4096, 2048 ,1024]:
                    for heads in [8 ,16]:
                        for d1 in [8,4,2,1]:
                            for d2 in [8,4,2,1]:
                                process(M,H,heads,mp_size,d1,d2,L,B)  
