'''
Here is to extract pretrained embeding from given files
'''


import torch
import numpy as np
from util import train_word
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#extract embedding
with open("data/all.review.vec.txt",'r') as f:
    pre_train = f.read().split('\n')

count = 0
miss = 0
embed_dim = 100
amazon_word_raw_dict = {}
amazon_wordlist = []
for i in range(1,len(pre_train)-1):
    a = pre_train[i].split(' ')
    word = a[0]
    #amazon_wordlist.append(word)
    del a[0]
    a.remove('')
    a=np.array([a],dtype = float)
    a = torch.from_numpy(a)
    a = a.float()
    amazon_word_raw_dict[word] = a




for i in range(len(train_word)):
    if train_word[i] in amazon_word_raw_dict:
        count = count + 1
        if i == 0:
            pre_embedding_weight = amazon_word_raw_dict[train_word[i]]
        else:
            pre_embedding_weight = torch.cat((pre_embedding_weight, amazon_word_raw_dict[train_word[i]]), 0)
    else:
        miss = miss + 1
        if i == 0:
            pre_embedding_weight = torch.rand(1, 100)
        else:
            pre_embedding_weight = torch.cat(      (     pre_embedding_weight,       torch.rand(1, 100)     ),0)
#pre_embedding_weight.todou
pre_embedding_weight = pre_embedding_weight.float()    
pre_embedding_weight = pre_embedding_weight.to(device)    
#embedding = nn.Embedding.from_pretrained(pre_embedding_weight)

