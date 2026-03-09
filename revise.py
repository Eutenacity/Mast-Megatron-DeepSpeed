new_list = []
with open("alog.txt")as f:
    lines=f.readlines()
    for line in lines:
        tmp=line.split(',')
        seq,m1,h1,heads1,mp1,dd1,dd2=int(tmp[3]),int(tmp[4]),int(tmp[5]),int(tmp[8]),int(tmp[9]),int(tmp[10]),int(tmp[11])
        new_list.append([float(tmp[0]),float(tmp[1]),seq,m1,h1,heads1,mp1,dd1,dd2])
        
output_file='out.txt'
with open("32configured_results.log")as f1:
    lines=f1.readlines()
    for line in lines:
        tmp=line.split(',')
        seq,m1,h1,heads1,mp1,dd1,dd2=int(tmp[3]),int(tmp[4]),int(tmp[5]),int(tmp[8]),int(tmp[9]),int(tmp[10]),int(tmp[11])
        
        
        flag = False
        for item in new_list:
            t1,t2,L,M,H,heads,mp_size,d1,d2 = item
            if seq==L and m1==M and h1==H and heads1==heads and mp1==mp_size and d1==dd1 and d2==dd2 and int(tmp[12])==0:
                print((seq,m1,h1,heads1,mp1,dd1,dd2))
                with open(output_file, "a+") as f:
                    f.write(
                        str(t1)
                        + ","
                        +str(t2)
                        + ","
                        + str(8)
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
                        + str('')
                        + "\n"
                    )
                flag=True
                break
        if not flag:
            with open(output_file, "a+") as f:
                f.write(line)
        