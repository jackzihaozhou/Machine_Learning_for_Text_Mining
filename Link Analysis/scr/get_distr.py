import numpy as np

def get_Pqt():
    with open('./data/query-topic-distro.txt','r') as f:
        distr_str = f.read()
        
    distr_raw = distr_str.split('\n')  
    
    #user->query->topic_index->distri
    Pqt = np.zeros((20,6,12,1), dtype='float')  
    
    for item in distr_raw[:-1]:
        item = item.split(" ")    
        for i in item[2:]:
            i = i.split(':')  
            #print(i)
            Pqt[int(item[0])-1][int(item[1])-1][int(i[0])-1] = float(i[1])
        
    return(Pqt)

def get_Put():
    with open('./data/user-topic-distro.txt','r') as f:
        distr_str = f.read()
        
    distr_raw = distr_str.split('\n')  
    
    #user->query->topic_index->distri
    Put = np.zeros((20,6,12,1), dtype='float')  
    
    for item in distr_raw[:-1]:
        item = item.split(" ")    
        for i in item[2:]:
            i = i.split(':')  
            #print(i)
            Put[int(item[0])-1][int(item[1])-1][int(i[0])-1] = float(i[1])
            
    return(Put)
    
    
#Put = get_Put()
#Pqt = get_Pqt()


def gqtd(Pqt,user,query,topic_index):
    return(Pqt[user-1][query-1][topic_index-1][0])
def gutd(Put,user,query,topic_index):
    return(Put[user-1][query-1][topic_index-1][0])
    
    
#get user_query

    
def get_user_query():
    with open('./data/query-topic-distro.txt','r') as f:
        distr_str = f.read()
    user = []
    query = []
    distr_raw = distr_str.split('\n')  
    for item in distr_raw[:-1]:
        item = item.split(" ")    
        user.append(int(item[0]))
        query.append(int(item[1]))
    return(user,query)
    
        #Pqt[int(item[0])-1][int(item[1])-1][int(i[0])-1] = float(i[1])
    