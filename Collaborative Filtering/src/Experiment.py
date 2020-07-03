import pandas as pd                 
import numpy as np       
import time
from Create_matrix import create_matrix


data = pd.read_csv("./data/train.csv",names=["MovieID","UserID","Rating","RatingDate"])
user_total = data["UserID"].unique().max() + 1#including 0
movie_total = data["MovieID"].unique().max() + 1 #including 0
dev_data = pd.read_csv("./data/dev.csv",names=["MovieID","UserID"])
dev_total = len(dev_data)
dev_movie = np.array(dev_data["MovieID"].tolist())
dev_user = np.array(dev_data["UserID"].tolist())


pro_user_user_matrix,pro_movie_movie_matrix,cos_user_user_matrix,cos_movie_movie_matrix = create_matrix()

def get_user_movie_rating_info():
    with open('./data/dev.queries','r') as f:
        tr_str = f.read()
    tr_raw = tr_str.split('\n')  
    dev_query_user_list = []
    user_movie_dict = {}
    for i in tr_raw[:-1]:
        movie_rating_dict = {}
        temp = i.split(' ')  
        userid = int(temp[0])
        dev_query_user_list.append(userid)
        #print(movie_rating_dict)
        if len(temp) > 1:
            for str in temp[1:]:
                movieid = int(str.split(':')[0])
                rating = float(str.split(':')[1])
                movie_rating_dict[movieid] = rating
            #print(movie_rating_dict)
            user_movie_dict[userid] = movie_rating_dict
        else:
            user_movie_dict[userid] = {}
    
    return(dev_query_user_list,user_movie_dict)       
    
#select top k relevant element form each row(excluding itself)
def find_user_topk(k,user_user_matrix,dev_query_user_list):
    #k =10
    user_topk ={}
    user_sim_value ={}
    for userid in dev_query_user_list:
        a = user_user_matrix[userid]
        b = np.array(np.unravel_index(np.argsort(a),a.shape))[0][::-1]
        sim_value= []
        topk = []
        count = 0
        for i in b:
            if count >= k:
                break
            
            if i == userid:
                continue
            
            count = count + 1
            topk.append(i)
            sim_value.append(a[i]) 
        user_topk[userid] = topk
        user_sim_value[userid] = sim_value
    return(user_topk,user_sim_value)
     

def get_rating_from_train_data(userid,movieid):
    temp = data[(data["UserID"] == userid )&( data["MovieID"] == movieid)]['Rating'].tolist()
    if len(temp)==0:
        rating = 0
    else:
        rating = temp[0]-3
    return(rating)
    
#useless    
def get_rating_from_dev_queries(userid,movieid,user_movie_dict):
    
    if movieid in user_movie_dict[userid]:
        rating = user_movie_dict[userid][movieid] - 3
    else:
        rating = 0
    return(rating)    
    


def mean_write_prediction(path,user_topk,k):
    mid_t = time.perf_counter()   
    global count
    global cache_hit_count
    f= open(path,"w")
    print("now write %s"%path)
    for index in range(dev_total):
        movieid = dev_movie[index]
        userid = dev_user[index]
        predict_rating = 0
        
        for simiar_userid in user_topk[userid]:
                
            rating = get_rating_from_train_data(simiar_userid,movieid)                
            predict_rating = predict_rating +(1/k)*rating            
        predict_rating = predict_rating + 3
        print("%f"%predict_rating,file = f)
    
    f.close()
    end_t = time.perf_counter()
    print("mean weight time: %f"%(end_t-mid_t))
    
def weighted_write_prediction(path,user_topk,user_sim_value,k):
    mid_t = time.perf_counter()   
    f= open(path,"w")
    print("now write %s"%path)
    for index in range(dev_total):
        movieid = dev_movie[index]
        userid = dev_user[index]
        predict_rating = 0
        
        i = 0
        abs_sum = sum(list(map(abs, user_sim_value[userid])))
        for simiar_userid in user_topk[userid]:
            
            rating = get_rating_from_train_data(simiar_userid,movieid)
            predict_rating = predict_rating +(user_sim_value[userid][i])*rating           
            i = i + 1
            
        predict_rating = (predict_rating/abs_sum) + 3
        if  abs_sum == 0:
            print("abus_sum = 0")
            predict_rating = 3
        if np.isnan(predict_rating):
            print("predict_rating is nan")
            predict_rating = 3
        print("%f"%predict_rating,file = f)
    
    f.close()
    end_t = time.perf_counter()
    print("mean weight time: %f"%(end_t-mid_t))
    
dev_query_user_list,user_movie_dict = get_user_movie_rating_info()

pro_user_topk_10,user_sim_value = find_user_topk(10,pro_user_user_matrix,dev_query_user_list)
mean_write_prediction("../eval/dev_pro_mean_useruser_10.txt",pro_user_topk_10,10)


pro_user_topk_100,user_sim_value = find_user_topk(100,pro_user_user_matrix,dev_query_user_list)
mean_write_prediction("../eval/dev_pro_mean_useruser_100.txt",pro_user_topk_100,100)

pro_user_topk_500,user_sim_value = find_user_topk(500,pro_user_user_matrix,dev_query_user_list)
mean_write_prediction("../eval/dev_pro_mean_useruser_500.txt",pro_user_topk_500,500)

cos_user_topk_10,user_sim_value = find_user_topk(10,cos_user_user_matrix,dev_query_user_list)
mean_write_prediction("../eval/dev_cos_mean_useruser_10.txt",cos_user_topk_10,10)
weighted_write_prediction("../eval/dev_cos_weighted_useruser_10.txt",cos_user_topk_10,user_sim_value,10)

cos_user_topk_100,user_sim_value = find_user_topk(100,cos_user_user_matrix,dev_query_user_list)
mean_write_prediction("../eval/dev_cos_mean_useruser_100.txt",cos_user_topk_100,100)
weighted_write_prediction("../eval/dev_cos_weighted_useruser_100.txt",cos_user_topk_100,user_sim_value,100)

cos_user_topk_500,user_sim_value = find_user_topk(500,cos_user_user_matrix,dev_query_user_list)
mean_write_prediction("../eval/dev_cos_mean_useruser_500.txt",cos_user_topk_500,500)
weighted_write_prediction("../eval/dev_cos_weighted_useruser_500.txt",cos_user_topk_500,user_sim_value,500)

