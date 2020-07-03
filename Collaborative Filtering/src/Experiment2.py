import pandas as pd  
import numpy as np       
import time
from Create_matrix import create_matrix

#global variable
data = pd.read_csv("./data/train.csv",names=["MovieID","UserID","Rating","RatingDate"])
user_total = data["UserID"].unique().max() + 1#including 0
movie_total = data["MovieID"].unique().max() + 1 #including 0

dev_data = pd.read_csv("./data/dev.csv",names=["MovieID","UserID"])
dev_total = len(dev_data)
dev_movie = np.array(dev_data["MovieID"].tolist())
dev_user = np.array(dev_data["UserID"].tolist())


pro_user_user_matrix,pro_movie_movie_matrix,cos_user_user_matrix,cos_movie_movie_matrix = create_matrix()

def find_movie_topk(k,movie_movie_matrix,dev_query_movie_list):
    movie_topk ={}
    movie_sim_value ={}
    for movieid in dev_query_movie_list:
        a = movie_movie_matrix[movieid]
        b = np.array(np.unravel_index(np.argsort(a),a.shape))[0][::-1]
        sim_value= []
        topk = []
        count = 0
        for i in b:
            if count >= k:
                break
            
            if i == movieid:
                continue
            
            count = count + 1
            topk.append(i)
            sim_value.append(a[i]) 
        movie_topk[movieid] = topk
        movie_sim_value[movieid] = sim_value
    return(movie_topk,movie_sim_value)
    
def get_rating_from_train_data(userid,movieid):
    temp = data[(data["UserID"] == userid )&( data["MovieID"] == movieid)]['Rating'].tolist()
    if len(temp)==0:
        rating = 0
    else:
        rating = temp[0]-3
    return(rating)
    
def mean_write_prediction(path,movie_topk,k):
    mid_t = time.perf_counter()   
    f= open(path,"w")
    print("now write %s"%path)
    for index in range(dev_total):
        movieid = dev_movie[index]
        userid = dev_user[index]
        predict_rating = 0
        
        for similar_movieid in movie_topk[movieid]:               
            rating = get_rating_from_train_data(userid,similar_movieid)                
            predict_rating = predict_rating +(1/k)*rating
            
        predict_rating = predict_rating + 3
        print("%f"%predict_rating,file = f)
    
    f.close()
    end_t = time.perf_counter()
    print("mean weight time: %f"%(end_t-mid_t))
    

def weighted_write_prediction(path,movie_topk,movie_sim_value,k):
    mid_t = time.perf_counter()   
    f= open(path,"w")
    print("now write %s"%path)
    for index in range(dev_total):
        movieid = dev_movie[index]
        userid = dev_user[index]
        predict_rating = 0
        abs_sum = sum(list(map(abs, movie_sim_value[movieid])))
        
        i = 0
        for similar_movieid in movie_topk[movieid]:
                
            rating = get_rating_from_train_data(userid,similar_movieid)                
            predict_rating = predict_rating +(movie_sim_value[movieid][i])*rating
            i = i + 1
            
        predict_rating = (predict_rating/abs_sum) + 3
        if abs_sum == 0:
            print("abs_sum = 0")
            predict_rating = 3
        if np.isnan(predict_rating):
            print("predict_rating is nan")
            predict_rating = 3
        print("%f"%predict_rating,file = f)
        
    f.close()
    end_t = time.perf_counter()
    print("weighted time: %f"%(end_t-mid_t))
    
dev_query_movie_list = dev_data["MovieID"].unique().tolist()
dev_query_movie_list.sort()


pro_movie_topk_10,movie_sim_value = find_movie_topk(10,pro_movie_movie_matrix,dev_query_movie_list)
mean_write_prediction("../eval/dev_pro_mean_moviemovie_10.txt",pro_movie_topk_10,10)


pro_movie_topk_100,movie_sim_value = find_movie_topk(100,pro_movie_movie_matrix,dev_query_movie_list)
mean_write_prediction("../eval/dev_pro_mean_moviemovie_100.txt",pro_movie_topk_100,100)

pro_movie_topk_500,movie_sim_value = find_movie_topk(500,pro_movie_movie_matrix,dev_query_movie_list)
mean_write_prediction("../eval/dev_pro_mean_moviemovie_500.txt",pro_movie_topk_500,500)


cos_movie_topk_10,movie_sim_value = find_movie_topk(10,cos_movie_movie_matrix,dev_query_movie_list)
mean_write_prediction("../eval/dev_cos_mean_moviemovie_10.txt",cos_movie_topk_10,10)
weighted_write_prediction("../eval/dev_cos_weighted_moviemovie_10.txt",cos_movie_topk_10,movie_sim_value,10)

cos_movie_topk_100,movie_sim_value = find_movie_topk(100,cos_movie_movie_matrix,dev_query_movie_list)
mean_write_prediction("../eval/dev_cos_mean_moviemovie_100.txt",cos_movie_topk_100,100)
weighted_write_prediction("../eval/dev_cos_weighted_moviemovie_100.txt",cos_movie_topk_100,movie_sim_value,100)

cos_movie_topk_500,movie_sim_value = find_movie_topk(500,cos_movie_movie_matrix,dev_query_movie_list)
mean_write_prediction("../eval/dev_cos_mean_moviemovie_500.txt",cos_movie_topk_500,500)
weighted_write_prediction("../eval/dev_cos_weighted_moviemovie_500.txt",cos_movie_topk_500,movie_sim_value,500)
