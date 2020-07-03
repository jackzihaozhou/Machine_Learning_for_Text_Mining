import numpy as np


class PMF(object):
    """PMF

    :param object:
    """

    def __init__(self, num_factors, num_users, num_movies):
        """__init__

        :param num_factors:
        :param num_users:
        :param num_movies:
        """
        # note that you should not modify this function
        np.random.seed(11)
        self.U = np.random.normal(size=(num_factors, num_users))
        self.V = np.random.normal(size=(num_factors, num_movies))
        self.num_users = num_users
        self.num_movies = num_movies

    def predict(self, user, movie):
        """predict

        :param user:
        :param movie:
        """
        # note that you should not modify this function
        return (self.U[:, user] * self.V[:, movie]).sum()

    def train(self, users, movies, ratings, alpha, lambda_u, lambda_v,
              batch_size, num_iterations):
        """train

        :param users: np.array of shape [N], type = np.int64
        :param movies: np.array of shape [N], type = np.int64
        :param ratings: np.array of shape [N], type = np.float32
        :param alpha: learning rate
        :param lambda_u:
        :param lambda_v:
        :param batch_size:
        :param num_iterations: how many SGD iterations to run
        """
        # modify this function to implement mini-batch SGD
        # for the i-th training instance,
        # user `users[i]` rates the movie `movies[i]`
        # with a rating `ratings[i]`.

        total_training_cases = users.shape[0]
        for i in range(num_iterations):
            start_idx = (i * batch_size) % total_training_cases
            users_batch = users[start_idx:start_idx + batch_size]
            movies_batch = movies[start_idx:start_idx + batch_size]
            ratings_batch = ratings[start_idx:start_idx + batch_size]
            curr_size = ratings_batch.shape[0]

            #My implementation of batch SGD here!!
            
            #set learning rate
            lr = 0.01
            
            #index corresponding to start from 0
            users_batch = users_batch -1
            movies_batch = movies_batch -1
            rate_batch = ratings_batch.reshape(-1,1)
            
            self.U = self.U.transpose()
            self.V = self.V.transpose()
            """
            calculate the batch results of (batch: Ui.T*V_j), here i and j is fixedly selected, i.e. the each elemnet in batch result column
            should be like (U[i].dot(V[j]), U[i+1].dot(V[j+1]), ....., U[i+curr_size].dot(V[j+curr_size])), in order to do this in matrix way,
            Do U[users_batch].dot(V[users_batch]),of with diagonal elemnets in result matrix are we want,
            extract them and make them as column vector
            """
            diagonal = np.diag_indices(curr_size)
            assert self.U[users_batch].shape[1]==self.V[movies_batch].shape[1],'latent factor not match'
            UV = np.dot(self.U[users_batch],self.V[movies_batch].transpose())
            UV = UV[diagonal].reshape(-1,1)
            #get grad_u and grad_v accoeding to the deriviation of eq. (4) in that paper
            grad_u = -(rate_batch-UV)*self.V[movies_batch] + lambda_u*self.U[users_batch]
            grad_v = -(rate_batch-UV)*self.U[users_batch] + lambda_v*self.V[movies_batch]
            
            #update correspoding batch row of U and V 
            self.U[users_batch] = self.U[users_batch] - lr*grad_u
            self.V[movies_batch] = self.V[movies_batch] - lr*grad_v
            
            
            self.U = self.U.transpose()
            self.V = self.V.transpose()
            
