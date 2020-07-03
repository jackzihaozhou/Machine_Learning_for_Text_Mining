import numpy as np


def doc_tp_dict():
    doc_tp = {}
    tp_doc = {}
    topic_list = []
    with open('./data/doc_topics.txt','r') as f:
        doc_topics_str = f.read()
    
    doc_topics_raw = doc_topics_str.split('\n')       
    
    for item in doc_topics_raw[:-1]:
        item = item.split(' ')
    
        if float(item[0]) not in doc_tp:        
            doc_tp[float(item[0])] = {float(item[1])}
        else:
            doc_tp[float(item[0])].add(float(item[1]))
        
        if float(item[1]) not in tp_doc:        
            tp_doc[float(item[1])] = {float(item[0])}
        else:
            tp_doc[float(item[1])].add(float(item[0]))
            
            
        if float(item[1]) not in topic_list:
            topic_list.append(float(item[1]))
    
    return(doc_tp,tp_doc)

def get_pt(t_index):    
    doc_tp,tp_doc = doc_tp_dict()
    
    doc_number = len(doc_tp)
    #print(len(tp_doc))
    pt = np.zeros(doc_number)
    
    #given topic t, calculate p_t     
    count = 0
    for i in range(1,doc_number+1): #traverse all doc
        if i in tp_doc[t_index]:
            count = count + 1
            pt[i-1]=1
            
    pt = pt.reshape((-1,1))/count
    return(pt)

  
