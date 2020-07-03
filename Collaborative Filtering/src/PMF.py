import pandas as pd                 
import numpy as np       
import torch
import time
import sys
from Create_matrix import create_pmf_matrix
#use gpu if we have
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")

rating_matrix = create_pmf_matrix()

ratings = pd.read_csv("./data/train.csv",names=["MovieID","UserID","Rating","RatingDate"])
dev_data = pd.read_csv("./data/dev.csv",names=["MovieID","UserID"])
#get data from dev
dev_total = len(dev_data)
dev_movie = np.array(dev_data["MovieID"].tolist())
dev_user = np.array(dev_data["UserID"].tolist())

n_users = ratings["UserID"].unique().max() + 1
n_movies = ratings["MovieID"].unique().max() + 1
min_rating, max_rating = ratings['Rating'].min(), ratings['Rating'].max()

def standarlized(rating_matrix):
    rating_matrix = (rating_matrix - min_rating) / (max_rating - min_rating)
    return(rating_matrix)

def get_rating(userid,movieid):
    predictions = torch.sigmoid(torch.mm(user_features[userid, :].view(1, -1), movie_features.t()))
    predicted_ratings = predictions.squeeze()
    return(predicted_ratings[movieid]+3)



class PMFLoss(torch.nn.Module):
    def __init__(self, lam_u, lam_v):
        super().__init__()
        self.lam_u = lam_u
        self.lam_v = lam_v
    
    def forward(self, matrix, u_features, v_features):
        predicted = torch.sigmoid(torch.mm(u_features, v_features.t()))
        
        diff = (matrix - predicted)*(matrix - predicted)
        err = torch.sum(diff)
        #equation 4 in the paper with lam_u and lam_v set to given value
        regularization = self.lam_u * torch.sum(u_features.norm(dim=1,p='fro')) + self.lam_v * torch.sum(v_features.norm(dim=1,p='fro'))
        
        loss = err + regularization
        return (loss)

rating_matrix = torch.from_numpy(rating_matrix)
rating_matrix = rating_matrix.to(device)
latent_vectors_list = [2,5,10,20,50]

for latent_vectors in latent_vectors_list:
    print("now we train PMF with latent_vectors = %d"%latent_vectors)
    user_features = torch.randn(n_users, latent_vectors, requires_grad=True,device=device)
    movie_features = torch.randn(n_movies, latent_vectors, requires_grad=True,device=device)
    #set both lam_u and lam
    pmferror = PMFLoss(lam_u=0.5, lam_v=0.5)
    optimizer = torch.optim.Adam([user_features, movie_features], lr=0.01)
    start_t = time.perf_counter()   
    iter = 1
    prev_loss = sys.float_info.max
    while(1):
        optimizer.zero_grad()
        loss = pmferror(rating_matrix, user_features, movie_features)
        loss.backward()
        optimizer.step()
        if iter % 50 == 0:
            print(f"iter {iter}, {loss:.3f}")
        iter = iter + 1
        
        if float(loss)>prev_loss or iter>50000:
            if(iter>50000):
                print("stop because iter>50000") 
            else:
                print("stop because converge") 
            break       
        prev_loss = float(loss)
        
    path = '../eval/rand1_PMF_latent_%d_lam0.5.txt'%latent_vectors
    f= open(path,"w")
    for index in range(dev_total):
        movieid = dev_movie[index]
        userid = dev_user[index]
        predicted_rating = get_rating(userid,movieid)
        print("%f"%predicted_rating,file=f)
    f.close()
    end_t = time.perf_counter()
    print("PMF training time for latent_vectors=%d: %f"%(latent_vectors,(end_t-start_t)))

