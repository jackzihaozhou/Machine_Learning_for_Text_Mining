# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:50:31 2020

@author: Jack
"""

import pandas as pd                 
import numpy as np       
import scipy.sparse as sparse
import scipy
import math

data = pd.read_csv("./data/train.csv",names=["MovieID","UserID","Rating","RatingDate"])

print("the total number of movies:")
print(len(data["MovieID"].unique()))
movie_total = data["MovieID"].unique().max() + 1
print("the total number of users:")
print(len(data["UserID"].unique()))
user_length = len(data["UserID"].unique())
user_total = data["UserID"].unique().max() + 1

rating_column = data["Rating"]
print("the number of times any movie was rated '1'")
rating1_column = rating_column[rating_column == 1]
print(len(rating1_column))

print("the number of times any movie was rated '3'")
rating3_column = rating_column[rating_column == 3]
print(len(rating3_column))

print("the number of times any movie was rated '5'")
rating5_column = rating_column[rating_column == 5]
print(len(rating5_column))

print("the average movie rating across all users and movies:")
print(rating_column.mean())


print("\nFor user ID 4321")
user4321 = data[data["UserID"] == 4321]
print("the number of movies rated:")
print(len(user4321["MovieID"].unique()))
print("the number of times the user gave a '1' rating")
print(len(user4321[user4321["Rating"]==1]))

print("the number of times the user gave a '3' rating")
print(len(user4321[user4321["Rating"]==3]))


print("the number of times the user gave a '5' rating")
print(len(user4321[user4321["Rating"]==5]))

print("the average movie rating for this user:")
print(user4321["Rating"].mean())
 
print("\nFormovie ID 3")
movie3 = data[data["MovieID"] == 3]
print("the number of users rating this movie")
print(len(movie3["UserID"].unique()))
print("the number of times the user gave a '1' rating")
print(len(movie3[movie3["Rating"]==1]))

print("the number of times the user gave a '3' rating")
print(len(movie3[movie3["Rating"]==3]))


print("the number of times the user gave a '5' rating")
print(len(movie3[movie3["Rating"]==5]))

print("the average rating for this movie:")
print(movie3["Rating"].mean())



print("\nNearest Neighbors for user4321:\n")

def find_top_k(list_a,k,id_list):
    #return a list of k index from big to small
    index_list = sorted(range(len(list_a)), key=lambda i: list_a[i])[-k:][::-1]
    toprank_list = []
    for index in index_list:
        toprank_list.append(id_list[index])
    return(toprank_list)

user4321_movie = np.array(user4321["MovieID"].tolist())
user4321_rating = np.array(user4321["Rating"].tolist())
row = np.zeros(len(user4321_movie))
column = user4321_movie #column index strat from 0
value = user4321_rating -3 #option 2
user4321_movie_vector = sparse.csc_matrix((value, (row, column)), shape=(1, movie_total)) #row vector
abs_user4321_value = math.sqrt((value*value).sum())

userid_list = []
product_similarity = []
cosine_similarity = []
for item in data["UserID"].unique():
    
    if(item == 4321):
        continue
    
    user_row = data[data["UserID"] == item]
    
    user_row_movie = np.array(user_row["MovieID"].tolist())
    
    user_row_rating = np.array(user_row["Rating"].tolist())
    
    row = np.zeros(len(user_row_movie))
    column = user_row_movie #column index strat from 0
    value = user_row_rating -3 #option 2
    abs_user_row_value = math.sqrt((value*value).sum())
    
    if(abs_user_row_value == 0):
        continue
    user_row_movie_vector_T = sparse.csc_matrix((value, (column, row)), shape=(movie_total, 1)) #switch row and column, column wector
    product = user4321_movie_vector.dot(user_row_movie_vector_T) #this result is 1*1 matrix, so "sum" lead to a number
    product = scipy.sparse.csr_matrix.sum(product)
    cos_product = product/(abs_user_row_value*abs_user4321_value)
    
    userid_list.append(item)
    product_similarity.append(product)
    cosine_similarity.append(cos_product)

#top 5
k = 5
product_similarity_rank = find_top_k(product_similarity,k,userid_list)
print("Top 5 NNs of user 4321 in terms of dot product similarity")
print(product_similarity_rank)
print("Top 5 NNs of user 4321 in terms of dot cosine similarity")
cosine_similarity_rank = find_top_k(cosine_similarity,k,userid_list)
print(cosine_similarity_rank)


movie3_user = np.array(movie3["UserID"].tolist())
movie3_rating = np.array(movie3["Rating"].tolist())
row = np.zeros(len(movie3_user))
column = movie3_user #column index strat from 0
value = movie3_rating -3 #option 2
movie3_user_vector = sparse.csc_matrix((value, (row, column)), shape=(1, user_total)) #row vector
abs_movie3_value = math.sqrt((value*value).sum())


print("\nNearest Neighbors for movie3:\n")


movieid_list = []
product_similarity = []
cosine_similarity = []
for item in data["MovieID"].unique():
    
    if item == 3:
        continue
    
    movie_row = data[data["MovieID"] == item]
    
    movie_row_user = np.array(movie_row["UserID"].tolist())
    
    movie_row_rating = np.array(movie_row["Rating"].tolist())
    
    row = np.zeros(len(movie_row_user))
    column = movie_row_user #column index strat from 0
    value = movie_row_rating -3 #option 2
    abs_movie_row_value = math.sqrt((value*value).sum())
    
    if(abs_movie_row_value == 0):
        continue
    movie_row_user_vector_T = sparse.csc_matrix((value, (column, row)), shape=(user_total, 1)) #switch row and column, column wector
    product = movie3_user_vector.dot(movie_row_user_vector_T) #this result is 1*1 matrix, so "sum" lead to a number
    product = scipy.sparse.csr_matrix.sum(product)
    cos_product = product/(abs_movie_row_value*abs_movie3_value)
    
    movieid_list.append(item)
    product_similarity.append(product)
    cosine_similarity.append(cos_product)   
    
k = 5
product_similarity_rank = find_top_k(product_similarity,k,movieid_list)
print("Top 5 NNs of movie3 in terms of dot product similarity")
print(product_similarity_rank)
print("Top 5 NNs of movie3 in terms of cosine similarity")
cosine_similarity_rank = find_top_k(cosine_similarity,k,movieid_list)
print(cosine_similarity_rank)






