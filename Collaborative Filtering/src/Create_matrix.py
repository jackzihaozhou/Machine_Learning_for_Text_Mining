import pandas as pd                 
import numpy as np       
import scipy.sparse as sparse
import math
import time
def create_matrix():
    print("start constructing matrix")
    data = pd.read_csv("./data/train.csv",names=["MovieID","UserID","Rating","RatingDate"])
    user_total = data["UserID"].unique().max() + 1#including 0
    movie_total = data["MovieID"].unique().max() + 1 #including 0
    
    row = np.array([])
    column = np.array([])
    cos_row = np.array([])
    cos_column = np.array([])
    
    pro_value = np.array([])
    cos_value = np.array([])
    start_t = time.perf_counter()
    for given_userid in data["UserID"].unique():#0-10915
    
        given_user = data[data["UserID"] == given_userid]
        
        given_user_user = np.array([given_userid for x in  range(len(given_user))])
        given_user_movie = np.array(given_user["MovieID"].tolist())
        given_user_rating = np.array(given_user["Rating"].tolist())
        row = np.concatenate((row,given_user_user))
        column = np.concatenate((column,given_user_movie))#column index strat from 0
        given_user_rating = given_user_rating -3
        pro_value = np.concatenate((pro_value,given_user_rating))
        
        abs_user_row = math.sqrt((given_user_rating*given_user_rating).sum())
        if(abs_user_row>0):
            abs_given_user_rating = given_user_rating/abs_user_row
            cos_row = np.concatenate((cos_row,given_user_user))
            cos_column = np.concatenate((cos_column,given_user_movie))#column index strat from 0
            cos_value = np.concatenate((cos_value,abs_given_user_rating))
    
    #product similarity
    pro_user_movie_matrix = sparse.csc_matrix((pro_value, (row, column)), shape=(user_total, movie_total)) #row vector
    
    
    pro_user_user_matrix = pro_user_movie_matrix.dot(pro_user_movie_matrix.transpose()) #none sparse
    pro_user_user_matrix = pro_user_user_matrix.toarray()
    
    pro_movie_movie_matrix = pro_user_movie_matrix.transpose().dot(pro_user_movie_matrix) #none sparse
    pro_movie_movie_matrix = pro_movie_movie_matrix.toarray()
    
    
    cos_user_movie_matrix = sparse.csc_matrix((cos_value, (cos_row, cos_column)), shape=(user_total, movie_total)) #row vector
    
    cos_user_user_matrix = cos_user_movie_matrix.dot(cos_user_movie_matrix.transpose()) #none sparse
    cos_user_user_matrix = cos_user_user_matrix.toarray()
    
    cos_movie_movie_matrix = cos_user_movie_matrix.transpose().dot(cos_user_movie_matrix) #none sparse
    cos_movie_movie_matrix = cos_movie_movie_matrix.toarray()
    
    mid_t = time.perf_counter()   
    print("matrix construction time: %f\n"%(mid_t-start_t)) 
    return(pro_user_user_matrix,pro_movie_movie_matrix,cos_user_user_matrix,cos_movie_movie_matrix)
    
def create_PCC_matrix():
    print("start constructing PCC movie matrix")
    data = pd.read_csv("./data/train.csv",names=["MovieID","UserID","Rating","RatingDate"])
    user_total = data["UserID"].unique().max() + 1#including 0
    movie_total = data["MovieID"].unique().max() + 1 #including 0
    
    row = np.array([])
    column = np.array([])
    cos_row = np.array([])
    cos_column = np.array([])
    
    pro_value = np.array([])
    cos_value = np.array([])
    for given_userid in data["UserID"].unique():#0-10915
    
        given_user = data[data["UserID"] == given_userid]
        
        given_user_user = np.array([given_userid for x in  range(len(given_user))])
        given_user_movie = np.array(given_user["MovieID"].tolist())
        given_user_rating = np.array(given_user["Rating"].tolist())
        given_user_rating = given_user_rating -3
        average = given_user_rating.sum()/len(given_user_rating)
        
        given_user_rating = given_user_rating - average # Centering
        L2_norm = math.sqrt((given_user_rating*given_user_rating).sum())

        if L2_norm == 0:
            print("L2_norm == 0")
            continue

        given_user_rating = given_user_rating/L2_norm
        row = np.concatenate((row,given_user_user))
        column = np.concatenate((column,given_user_movie))#column index strat from 0
        pro_value = np.concatenate((pro_value,given_user_rating))
        
        abs_user_row = math.sqrt((given_user_rating*given_user_rating).sum())
        if(abs_user_row>0):
            abs_given_user_rating = given_user_rating/abs_user_row
            cos_row = np.concatenate((cos_row,given_user_user))
            cos_column = np.concatenate((cos_column,given_user_movie))#column index strat from 0
            cos_value = np.concatenate((cos_value,abs_given_user_rating))
    
        #product similarity
    pro_user_movie_matrix = sparse.csc_matrix((pro_value, (row, column)), shape=(user_total, movie_total)) #row vector
    
    pro_movie_movie_matrix = pro_user_movie_matrix.transpose().dot(pro_user_movie_matrix) #none sparse
    pro_movie_movie_matrix = pro_movie_movie_matrix.toarray()
    
    
    cos_user_movie_matrix = sparse.csc_matrix((cos_value, (cos_row, cos_column)), shape=(user_total, movie_total)) #row vector
    
    
    cos_movie_movie_matrix = cos_user_movie_matrix.transpose().dot(cos_user_movie_matrix) #none sparse
    cos_movie_movie_matrix = cos_movie_movie_matrix.toarray()
    
    return(pro_movie_movie_matrix,cos_movie_movie_matrix)
    

    
def create_pmf_matrix():
    print("start constructing PMF matrix")
    data = pd.read_csv("./data/train.csv",names=["MovieID","UserID","Rating","RatingDate"])
    user_total = data["UserID"].unique().max() + 1#including 0
    movie_total = data["MovieID"].unique().max() + 1 #including 0
    
    row = np.array([])
    column = np.array([])    
    pro_value = np.array([])
    for given_userid in data["UserID"].unique():#0-10915
    
        given_user = data[data["UserID"] == given_userid]
        
        given_user_user = np.array([given_userid for x in  range(len(given_user))])
        given_user_movie = np.array(given_user["MovieID"].tolist())
        given_user_rating = np.array(given_user["Rating"].tolist())
        row = np.concatenate((row,given_user_user))
        column = np.concatenate((column,given_user_movie))#column index strat from 0
        given_user_rating = given_user_rating - 3
        pro_value = np.concatenate((pro_value,given_user_rating))
    
    #product similarity

    pro_user_movie_matrix = sparse.csc_matrix((pro_value, (row, column)), shape=(user_total, movie_total)) #row vector
    user_movie_matrix = pro_user_movie_matrix.toarray()
    return(user_movie_matrix)