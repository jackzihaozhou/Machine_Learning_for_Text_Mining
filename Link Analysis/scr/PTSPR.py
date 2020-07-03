import numpy as np
import os
#import scipy as sp 
#import scipy.sparse as sparse
import time
from trans_matrix import trans_matrix
from get_distr import get_Put,gutd,get_user_query
from extract_score import load_result, load_score
from get_pt import get_pt

def PTSPR():
    #r_prev = np.zeros(length).reshape((-1,1))/length
    M_T , doc_length, row_miss = trans_matrix()
    r_initial = np.ones(doc_length).reshape((-1,1))/doc_length
    
    
    #WS
    
    alpha = 0.8
    weight = 0.999
    
    user,query = get_user_query()
    start_t = time.perf_counter()  
    #topic range: [1,12] for topic:
    r = []
    for topic_id in range(1,13):
        #topic_id = 1
        pt =  get_pt(topic_id)
        difference = 1
        #print("start calculating rt:")
        rt = r_initial
        
        
        r_assist = np.ones(doc_length).reshape((-1,1))/doc_length
        while(difference>1e-8):
            prev_r = rt
            sum = 0
            for i in row_miss:
                sum = sum + rt[i-1]
            rt= alpha*M_T.dot(rt)+(1-alpha)*pt + alpha*sum*r_assist
    
            r_diff = rt - prev_r
            difference = abs(r_diff).sum()
    
        r.append(rt)
        
    mid_t = time.perf_counter()       
    Put = get_Put()    
    
    root = './'
    file = f'PTSPR-WS.txt'
    path = os.path.join(root,file)
    f_QSTPR = open(path,'w')
    f = open("./PTSPR-U2Q1.txt","w")
    #for user:
            
    for i in range(len(user)):
        user_id = user[i]  
        query_id = query[i]
        # now start calculate rq : QTSPR #traverse topic_id
        
        rq = 0
        for topic in range(12):#0-11
            utd = gutd(Put,user_id,query_id,topic)
            rq = rq + utd *r[topic]
        
    
        
        
        doc_score = {}
        doc_IRscore = load_result(user_id,query_id)
        #origin = []
        
        
        for doc_num in doc_IRscore:
            
            IRscore = load_score(doc_IRscore,doc_num)
            PRscore = rq[doc_num-1]#doc_num-1 = rt_doc_index
            score = weight * PRscore + (1-weight) * IRscore
            doc_score[doc_num] = score
            #origin.append(doc_num)
        
        
        
        
        rank = sorted(doc_score.items(), key=lambda item:item[1], reverse=True)
        for i in range(len(rank)):
            print("%d-%d Q0 %d %d %f PTSPR-WS"%(user_id,query_id,rank[i][0],i+1,rank[i][1]),file = f_QSTPR)
            if(user_id == 2 and query_id == 1):
                print("%d-%d Q0 %d %d %f PTSPR"%(user_id,query_id,rank[i][0],i+1,rank[i][1]),file = f)

                
    end_t = time.perf_counter()   
    
    print("\nPTSPR-WS: %f secs for PageRank, %f secs for retrieval"%(mid_t-start_t,end_t-mid_t))
    
    f_QSTPR.close()
    f.close()
    
    #NS
    
    user,query = get_user_query()
    start_t = time.perf_counter()  
    #topic range: [1,12] for topic:
    r = []
    for topic_id in range(1,13):
        #topic_id = 1
        pt =  get_pt(topic_id)
        difference = 1
        #print("start calculating rt:")
        rt = r_initial
        
        
        r_assist = np.ones(doc_length).reshape((-1,1))/doc_length
        while(difference>1e-8):
            prev_r = rt
            sum = 0
            for i in row_miss:
                sum = sum + rt[i-1]
            rt= alpha*M_T.dot(rt)+(1-alpha)*pt + alpha*sum*r_assist
    
            r_diff = rt - prev_r
            difference = abs(r_diff).sum()
            #print(rt)
        r.append(rt)
        
    mid_t = time.perf_counter()       
    
    Put = get_Put()    
    
    root = "./"
    file = f'PTSPR-NS.txt'
    path = os.path.join(root,file)
    f_QSTPR = open(path,'w')
    #for user:
            
    for i in range(len(user)):
        user_id = user[i]  
        query_id = query[i]
        # now start calculate rq : QTSPR #traverse topic_id
        
        rq = 0
        for topic in range(12):#0-11
            utd = gutd(Put,user_id,query_id,topic)
            rq = rq + utd *r[topic]
        
    
        
        
        doc_score = {}
        doc_IRscore = load_result(user_id,query_id)
        
        
        
        for doc_num in doc_IRscore:
            
            #IRscore = load_score(doc_IRscore,doc_num)
            PRscore = rq[doc_num-1]#doc_num-1 = rt_doc_index
            score = PRscore
            doc_score[doc_num] = score
            
        
        
        
        
        rank = sorted(doc_score.items(), key=lambda item:item[1], reverse=True)
        for i in range(len(rank)):
            print("%d-%d Q0 %d %d %f PTSPR-NS"%(user_id,query_id,rank[i][0],i+1,rank[i][1]),file = f_QSTPR)
            
    end_t = time.perf_counter()   
    
    print("\nPTSPR-NS: %f secs for PageRank, %f secs for retrieval"%(mid_t-start_t,end_t-mid_t))
    
    f_QSTPR.close()
    
    
    
    #CM
    
    user,query = get_user_query()
    start_t = time.perf_counter()  
    #topic range: [1,12] for topic:
    r = []
    for topic_id in range(1,13):
        #topic_id = 1
        pt =  get_pt(topic_id)
        difference = 1
        #print("start calculating rt:")
        rt = r_initial
        
        
        r_assist = np.ones(doc_length).reshape((-1,1))/doc_length
        while(difference>1e-8):
            prev_r = rt
            sum = 0
            for i in row_miss:
                sum = sum + rt[i-1]
            rt= alpha*M_T.dot(rt)+(1-alpha)*pt + alpha*sum*r_assist
    
            r_diff = rt - prev_r
            difference = abs(r_diff).sum()
            #print(rt)
        r.append(rt)
        
    mid_t = time.perf_counter()       
    
    Put = get_Put()    
    
    root = "./"
    file = f'PTSPR-CM.txt'
    path = os.path.join(root,file)
    f_QSTPR = open(path,'w')
    #for user:
            
    for i in range(len(user)):
        user_id = user[i]  
        query_id = query[i]
        # now start calculate rq : QTSPR #traverse topic_id
        
        rq = 0
        for topic in range(12):#0-11
            utd = gutd(Put,user_id,query_id,topic)
            rq = rq + utd *r[topic]
        
        #MINMAX SCALER
        rq_min = rq.min()
        rq_max = rq.max()    
        rq = (rq-rq_min)/(rq_max-rq_min)
        
        
        doc_IRscore_temp = load_result(user_id,query_id)
        doc_IRscore_value = np.array(list(doc_IRscore_temp.values()))
        doc_IRscore_key = np.array(list(doc_IRscore_temp.keys()))   
        doc_IRscore_value_min = min(doc_IRscore_value)
        doc_IRscore_value_max = max(doc_IRscore_value)    
        doc_IRscore_value = (doc_IRscore_value - doc_IRscore_value_min)/(doc_IRscore_value_max - doc_IRscore_value_min)
            
        doc_IRscore = {}
        for i in range(len(doc_IRscore_key)):
            doc_IRscore[doc_IRscore_key[i]] = doc_IRscore_value[i]
        
        
        doc_score = {}
        weight = 0.5
        for doc_num in doc_IRscore:
            
            
            IRscore = load_score(doc_IRscore,doc_num)
            PRscore = rq[doc_num-1]#doc_num-1 = rt_doc_index
            score = weight * PRscore + (1-weight) * IRscore
            doc_score[doc_num] = score
        
        
        
        rank = sorted(doc_score.items(), key=lambda item:item[1], reverse=True)
        for i in range(len(rank)):
            print("%d-%d Q0 %d %d %f PTSPR-CM"%(user_id,query_id,rank[i][0],i+1,rank[i][1]),file = f_QSTPR)
            
    end_t = time.perf_counter()   
    
    print("\nPTSPR-CM: %f secs for PageRank, %f secs for retrieval"%(mid_t-start_t,end_t-mid_t))
    
    f_QSTPR.close()