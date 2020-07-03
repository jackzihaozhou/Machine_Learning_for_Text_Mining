import sys
import numpy as np
from scipy.sparse import csr_matrix,hstack
from numpy.random import default_rng
from sklearn.datasets import load_svmlight_file
import time
import os

def mkdir(name):
    """this function is to create a directory, if there is no directory named "name", create it, otherwise, do nothing

    Args:
        :param name: the directory name you want to create with prefix path

    Returns:
        void

    """
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name) 

def eval(iter,w,grad_norm,X_test,y_test,X_train, y_train,f_opt,time_used,output_file):   
    """this function will evaluate the current w from current trainig procedure, it will evaluate:
        :test_accuracy
        :grad_norm
        :optimal function value on train set
        :optimal function value on test set
        :relative function value difference of test
        :current time used
    Args:
        :param iter: current iter
        :param w: current w
        :param grad_norm: current grad_norm
        :param X_test: X_test
        :param y_test: y_test
        :param X_train: X_train
        :param y_train: y_train
        :param f_opt: optimal function value in table 1
        :param grad_norm: current grad_norm
        :param time_used: time duration from start time to current time
        :param output_file: the file you want to store the eval result
    Returns:
        void
    """
    predict = X_test.dot(w).toarray()
    predict = np.bool_(predict>0)
    label = np.bool_(y_test.toarray()>0)
    right = sum(predict.squeeze()==label.squeeze())
    test_accuracy = right/X_test.shape[0]
    
    
  
    judge = 1 - y_test.multiply(X_test.dot(w)).toarray()
    index_test_I = np.argwhere(judge.squeeze()>0).squeeze()
    max2 = judge[index_test_I]*judge[index_test_I]
    f_test_second = np.sum(max2)*(2*lamda/y_test.shape[0])
    f_test_first = w.transpose().dot(w).toarray()[0][0]
    f_test = 0.5*f_test_first + f_test_second

    
    rng = default_rng()
    ranmdon_index = rng.choice(n, size=int(n/5), replace=False)
    
    X_val = X_train[ranmdon_index]
    y_val = y_train[ranmdon_index].reshape(-1,1) 
    
    judge = 1 - y_val.multiply(X_val.dot(w)).toarray()
    index_val_I = np.argwhere(judge.squeeze()>0).squeeze()
    max2 = judge[index_val_I]*judge[index_val_I]
    f_val_second = np.sum(max2)*(2*lamda/y_val.shape[0])
    f_val_first = w.transpose().dot(w).toarray()[0][0]
    f_val = 0.5*f_val_first + f_val_second
    
    f_rel_diff = (f_val-f_opt)/f_val
    #print(f"iter: {iter} test_accuracy: {test_accuracy} train_accuracy: {train_accuracy} grad_norm: {grad_norm} val_f: {f_val} test_f*: {f_test} f_rel_diff: {f_rel_diff} time_used: {time_used}")
    #print(f"iter: {iter}\ntest_accuracy: {test_accuracy}\ngrad_norm: {grad_norm}\nval_f: {f_val}\ntest_f*: {f_test}\nf_rel_diff: {f_rel_diff}\ntime_used: {time_used}s")
    print(f"iter: {iter} test_accuracy: {test_accuracy} time_used: {time_used}")
    print(f"iter: {iter} test_accuracy: {test_accuracy} grad_norm: {grad_norm} val_f: {f_val} test_f*: {f_test} f_rel_diff: {f_rel_diff} time_used: {time_used}",file = output_file)
#---------------------------------------------------------------------------------------------------------------------------#    
    
# read the train file from first arugment
train_file = sys.argv[1]

# read the test file from second argument
test_file = sys.argv[2]

# modify data and add bias column
X_train, y_train = load_svmlight_file(train_file)
X_test, y_test = load_svmlight_file(test_file)

y_train = csr_matrix(y_train.reshape(-1,1))
y_test = csr_matrix(y_test.reshape(-1,1))

bias_train_column = np.ones(X_train.shape[0]).reshape(-1,1)
bias_train_column = bias_train_column*0.01
bias_train_column = csr_matrix(bias_train_column)
X_train = hstack((X_train,bias_train_column))

bias_test_column = np.ones(X_test.shape[0]).reshape(-1,1)
bias_test_column = bias_test_column*0.01
bias_test_column = csr_matrix(bias_test_column)
X_test = hstack((X_test,bias_test_column))

y_test = y_test.reshape(-1,1)
n = X_train.shape[0]

X_train = X_train.tocsr()
X_test = X_test.tocsr()

# select hyperparameter according to given dataset
if(n==57847):
    lr_init = 0.001
    lamda = 7230.875
    epoch_max = 50
    f_opt = 669.664812
    file_name="result/sgd_output_real.txt"
    batch_size = 1000
    beta = 0
else:
    lr_init = 0.0001
    lamda = 3631.3203125
    epoch_max = 40
    f_opt = 2541.664519
    file_name="result/sgd_output_cov.txt"
    batch_size = 10000
    beta = 0.1

#initialization w as column vector
w = np.random.uniform(low=-1.0, high=1.0, size=(1,X_train.shape[1])).reshape(-1,1)
w = csr_matrix(0.1*w)

#creat a directory named "result" to store the result
mkdir("result")
output_file = open(file_name,"w")

print("\nstart sgd training\n")
print(f"the result will be stored in: {file_name}")

#start training
start_t = time.perf_counter()   
iter = 0
for eph in range(1,epoch_max+1):
    print(f"eph:{eph}")
    start_idx = 0
    while(1):
    
        if(start_idx + batch_size > n):
            break
        #update annealing learning rate
        lr = lr_init/(1+beta*eph)
        
        X_batch = X_train[start_idx:start_idx + batch_size]
        y_batch = y_train[start_idx:start_idx + batch_size]
        
        
        multiply_result = X_batch.dot(w)            
        judge = y_batch.multiply(multiply_result)
        judge = judge.toarray()
        index_I = np.argwhere(judge.squeeze()<1).squeeze()
        a = X_batch[index_I].dot(w)- y_batch[index_I]
        b = X_batch[index_I].transpose().dot(a)
        grad = b.multiply(2*lamda/X_batch.shape[0]) + w
        w = w - grad.multiply(lr)
        start_idx = start_idx + batch_size
        iter = iter + 1
    
        grad_norm=np.linalg.norm(grad.toarray(), ord=2, axis=None, keepdims=False)
        
        temp_t = time.perf_counter()
        time_used = temp_t-start_t
        if iter%20 == 0:
            eval(iter,w,grad_norm,X_test,y_test,X_train, y_train,f_opt,time_used,output_file)
    
end_t = time.perf_counter()   
print(f"total time used*: {end_t-start_t}")

output_file.close()
