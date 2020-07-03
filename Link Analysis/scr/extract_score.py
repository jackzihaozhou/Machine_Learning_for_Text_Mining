import os

def load_result(user_id,query_id):

#user_id = 2
#query_id = 1
    root = './'
    folder = 'data/indri-lists/'
    file = f'{user_id}-{query_id}.results.txt'
    
    path = os.path.join(root,folder,file)
    with open(path,'r') as f:
        result_str = f.read()
        
    result = result_str.split('\n')  
    
    doc_score = {}
    
    for item in result[:-1]:
        item = item.split(" ") 

        if int(item[2]) not in doc_score:        
            doc_score[int(item[2])] = float(item[4])
        else:
            print("impossible")
    #print(doc_score)
    return(doc_score)
        


doc_score = load_result(2,1)

def load_score(doc_score,doc_num):
    if doc_num not in doc_score:
        return(-10)
    else:
        return(doc_score[doc_num])
    
#IRscore = load_score(doc_score,11111)

