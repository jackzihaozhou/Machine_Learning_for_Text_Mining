'''
Here are the data processing and untility function for CNN and RNN 
'''


import os
import torch
import numpy as np
from torch.utils import data

#from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_prefix = "./data/train/"
test_prefix = "./data/test/"
positive_prefix = "positive/"
negative_prefix = "negative/"
train_prefix = "./data/train/"
test_prefix = "./data/test/"
positive_prefix = "positive/"
negative_prefix = "negative/"
train_positive = os.path.join(train_prefix, positive_prefix)
train_negative = os.path.join(train_prefix, negative_prefix)
test_positive = os.path.join(test_prefix, positive_prefix)
test_negative = os.path.join(test_prefix, negative_prefix)
train_loc = [train_positive,train_negative]
test_loc = [test_positive,test_negative]


train_doc_num = 0
train_total_length = 0
train_max_length = 0
train_word2count = {}
train_idx2sentence = {}
train_idx2label = {}

test_doc_num = 0
test_idx2sentence = {}
test_idx2label = {}


def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 


def parse_document(doc_str,label,sign="train"):
    if sign == "train":
        
        global train_doc_num
        global train_total_length
        global train_max_length
        global train_idx2sentence
        global train_word2count
        global train_idx2label
        
        train_idx2sentence[train_doc_num] = doc_str
        train_idx2label[train_doc_num] = label
        train_doc_num = train_doc_num + 1
        
        doc_word = doc_str.split(' ')  
        train_total_length = train_total_length + len(doc_word)
        if len(doc_word) > train_max_length:
            train_max_length = len(doc_word)
            
        
        for word in doc_word:
            
            if word not in train_word2count:
                train_word2count[word] = 1                     
            else:
                train_word2count[word] = train_word2count[word] + 1
                
    elif sign == "test":
        
        global test_doc_num
        global test_idx2sentence
        global test_idx2label
        
        test_idx2sentence[test_doc_num] = doc_str
        test_idx2label[test_doc_num] = label
        test_doc_num = test_doc_num + 1
        
        doc_word = doc_str.split(' ')  

    else:
        print("wrong input parameter")
        
        
        

for loc in train_loc:
    for root, dirs, files in os.walk(loc):
        for name in files:
            with open(os.path.join(root, name),'r') as f:

                doc_str = f.read()
                if "positive" in root:
                    parse_document(doc_str,1,"train")
                else:
                    parse_document(doc_str,0,"train")
                    
train_word2count_order=sorted(train_word2count.items(),key=lambda x:x[1],reverse=True) 

for loc in test_loc:
    for root, dirs, files in os.walk(loc):
        for name in files:
            with open(os.path.join(root, name),'r') as f:

                doc_str = f.read()
                if "positive" in root:
                    parse_document(doc_str,1,"test")
                else:
                    parse_document(doc_str,0,"test")



def get_wordlist(max_number):
    train_word = []
    index2word = {}
    word2index = {}
    
    index2word[0] = "<pad>"
    word2index["<pad>"] = 0 
    train_word.append("<pad>")
    
    index2word[1] = "<unk>"
    word2index["<unk>"] = 1
    train_word.append("<unk>")
    
    for idx in range(2,max_number):
        index2word[idx] = train_word2count_order[idx-2][0]
        word2index[train_word2count_order[idx-2][0]] = idx 
        train_word.append(train_word2count_order[idx-2][0])
    return(index2word,word2index,train_word)
    


def get_idx_from_word(word):
    if word in word2index:
        return(word2index[word])
    else:
        return(1) #"<unk>"




index2word,word2index,train_word = get_wordlist(10000)


max_length = 300
train = np.zeros((len(train_idx2sentence),max_length),dtype=np.int32)
#build tensor for sentence(train)
for idx in train_idx2sentence:
    word = train_idx2sentence[idx].split(' ')
    length = len(word)
    for i in range(max_length):
        if i<length:
            train[idx][i] =  get_idx_from_word(word[i])
        else:
            train[idx][i] = 0
            
            
train_tensor = torch.from_numpy(train)
train_tensor = train_tensor.long()
train_tensor = train_tensor.to(device)            
            
            
train_label = np.zeros(len(train_idx2label),dtype=np.float32)
for idx in train_idx2label:
    train_label[idx] = train_idx2label[idx]
train_labels = torch.from_numpy(train_label)
train_labels = train_labels.float()
train_labels = train_labels.to(device)


test = np.zeros((len(test_idx2sentence),max_length),dtype=np.int32)
#build tensor for sentence(test)
for idx in test_idx2sentence:
    word = test_idx2sentence[idx].split(' ')
    length = len(word)
    for i in range(max_length):
        if i<length:
            test[idx][i] =  get_idx_from_word(word[i])
        else:
            test[idx][i] = 0 #"<pad>"
            
            
test_tensor = torch.from_numpy(test)
test_tensor = test_tensor.long()
test_tensor = test_tensor.to(device)            
            
            
test_label = np.zeros(len(test_idx2label),dtype=np.float32)
for idx in test_idx2label:
    test_label[idx] = test_idx2label[idx]
test_labels = torch.from_numpy(test_label)
test_labels = test_labels.float()
test_labels = test_labels.to(device)



class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.len = len(images)

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)



def get_traindataloader(batch):
    dataset = MyDataset(train_tensor, train_labels)
    train_generator = data.DataLoader(dataset, batch_size=batch, shuffle=True)
    return(train_generator)
    
def get_testdataloader(batch):
    dataset = MyDataset(test_tensor, test_labels)
    test_generator = data.DataLoader(dataset, batch_size=batch, shuffle=False)
    return(test_generator)

def output_statistic():
    print(f"the total number of unique words in T: {len(train_word2count)}")
    print(f"the total number of training examples in T: {train_doc_num}")
    print("the ratio of positive examples to negative examples in T: 1")
    print(f"the average length of document in T: {train_total_length/train_doc_num}")
    print(f"the max length of document in T: {max_length}")