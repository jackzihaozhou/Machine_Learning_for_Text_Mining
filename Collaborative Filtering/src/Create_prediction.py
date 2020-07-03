# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 03:09:23 2020

@author: Jack
"""


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

test_data = pd.read_csv("./data/test.csv",names=["MovieID","UserID"])
test_total = len(test_data)
test_movie = np.array(test_data["MovieID"].tolist())
test_user = np.array(test_data["UserID"].tolist())


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
        if len(temp) > 1:
            for str in temp[1:]:
                movieid = int(str.split(':')[0])
                rating = float(str.split(':')[1])
                movie_rating_dict[movieid] = rating
            user_movie_dict[userid] = movie_rating_dict
        else:
            user_movie_dict[userid] = {}
    
    return(dev_query_user_list,user_movie_dict)
def get_test_user_movie_rating_info():
    with open('./data/test.queries','r') as f:
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
    user_topk ={}
    user_sim_value ={}
    for userid in dev_query_user_list:
        #userid = 1
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
     
def get_rating_from_dev_queries(userid,movieid,user_movie_dict):
    
    if movieid in user_movie_dict[userid]:
        rating = user_movie_dict[userid][movieid] - 3
    else:
        rating = 0
    return(rating)    
    


def mean_write_prediction(path,user_topk,k,mode = 'd'):
    mid_t = time.perf_counter()   
    global count
    global cache_hit_count
    f= open(path,"w")
    print("now write %s"%path)
    for index in range(dev_total):
        if mode == 'd':
            movieid = dev_movie[index]
            userid = dev_user[index]
        else:
            
            movieid = test_movie[index]
            userid = test_user[index]
            #print(userid)
        predict_rating = 0
        
        for simiar_userid in user_topk[userid]:
            rating = get_rating_from_train_data(simiar_userid,movieid)                
            predict_rating = predict_rating +(1/k)*rating
            
        predict_rating = predict_rating + 3
        print("%f"%predict_rating,file = f)
    
    f.close()
    end_t = time.perf_counter()
    print("mean weight time: %f"%(end_t-mid_t))
    
dev_query_user_list,user_movie_dict = get_user_movie_rating_info()
test_query_user_list,user_movie_dict = get_test_user_movie_rating_info()

pro_user_topk_10,user_sim_value = find_user_topk(10,pro_user_user_matrix,test_query_user_list)
mean_write_prediction("../test-predictions.txt",pro_user_topk_10,10,mode='t')
pro_user_topk_10,user_sim_value = find_user_topk(10,pro_user_user_matrix,dev_query_user_list)
mean_write_prediction("../dev-predictions.txt",pro_user_topk_10,10)




