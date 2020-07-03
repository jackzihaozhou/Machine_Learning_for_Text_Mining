def trans_matrix():

    import numpy as np
    #import scipy as sp 
    import scipy.sparse as sparse
    
    #print (sp.version)
    row =[]   
    column = []
    value = []
    
    with open('./data/transition.txt','r') as f:
        tr_str = f.read()
        
    tr_raw = tr_str.split('\n')   
    dict_row = {}
    row_miss = []
    for item in tr_raw[:-1]:
        item = item.split(' ')
        #print(item)
        #float(item[0]) is row_index 
        if int(item[0]) not in dict_row:        
            dict_row[int(item[0])] = 1
        else:
            dict_row[int(item[0])] = dict_row[int(item[0])] + 1
            
        row.append(int(item[0]))
        column.append(int(item[1]))
        value.append(int(item[2]))        
    
            
            
            
    for i in range(len(value)):
        #print(dict_row[row[i]])
        value[i]= value[i]/(dict_row[row[i]])
        
    for i in range(1,81434):#1-81433
        if i not in dict_row:
            row_miss.append(i)

    #print(raw_miss)
    #np.zeros(len(raw_miss)*81433)
    #coo_matrix是最简单的存储方式。采用三个数组row、col和data保存非零元素的信息。这三个数组的长度相同，row保存元素的行，col保存元素的列，data保存元素的值。
    #一般来说，coo_matrix主要用来创建矩阵，因为coo_matrix无法对矩阵的元素进行增删改等操作，一旦矩阵创建成功以后，会转化为其他形式的矩阵。data = [5,2,3,0]
    
    row  = np.array(row)
    column  = np.array(column)
    
    length = max(row.max(),column.max())
    
    #scipy index strat from 0, not 1
    row = row -1
    column = column -1
    data = np.array(value)
    
    #M = sp.sparse.csc_matrix((data, (row, column)), shape=(length, length))
    M_T = sparse.csc_matrix((data, (column, row)), shape=(length, length))
    
    #r_prev = np.zeros(length).reshape((-1,1))/length

    return(M_T,length,row_miss)


#M_T,length,raw_miss = trans_matrix()