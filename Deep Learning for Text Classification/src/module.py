'''
Module structure of CNN and RNN
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
 
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                embedding_dim=config.embedding_size)

        self.convs = nn.Sequential(nn.Conv1d(in_channels = config.embedding_size, 
                                        out_channels = config.filter_num, 
                                        kernel_size = config.filter_length),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=config.maxpool_kernel,padding=1))
              
        self.linear = nn.Linear(in_features = (config.filter_num * int((config.sentence_length)/config.maxpool_kernel)), out_features =config.num_class)#1 positive or negetive
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if self.config.pre_embed == False:
            embed_x = self.embedding(x)
        else:
            from pretrain_weight import pre_embedding_weight
            pretrained_embedding = nn.Embedding.from_pretrained(pre_embedding_weight)        
            embed_x = pretrained_embedding(x)  
            
        embed_x = embed_x.permute(0, 2, 1) 
        out = self.convs(embed_x)  
        out = out.permute(0,2,1)
        out = out.reshape([self.config.batch_size,-1,1])
        out = torch.squeeze(out)
        out = F.dropout(input=out, p=0.3)
        out = self.linear(out)
        out = torch.squeeze(out)
        out  = self.sigmoid(out)
        return out
    
class CNNconfiguration():    
    def __init__(self,vocab_size,embedding_size,filter_num,filter_length,num_class,sentence_length,maxpool_kernel,batch_size,pre_embed = False):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_num = filter_num
        self.filter_length = filter_length
        self.num_class = num_class
        self.sentence_length = sentence_length
        self.maxpool_kernel = maxpool_kernel
        self.batch_size = batch_size
        self.pre_embed = pre_embed
        
        
class TextRNN(nn.Module):
   def __init__(self, config):      
        super(TextRNN, self).__init__()
       
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                embedding_dim=config.embedding_size)
     
        self.lstm = nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size,num_layers=config.num_layers, batch_first = True)
        self.tanh = nn.Tanh()
        
        self.maxpool = nn.MaxPool1d(kernel_size=config.sentence_length,padding=1)

        self.linear = nn.Linear(in_features = config.hidden_size,#100
                                out_features =config.num_class)#1 positive or negetive

        self.sigmoid = nn.Sigmoid()
        
   def forward(self, x):
       
        if self.config.pre_embed == False:
            embed_x = self.embedding(x)
        else:
            from pretrain_weight import pre_embedding_weight
            pretrained_embedding = nn.Embedding.from_pretrained(pre_embedding_weight)        
            embed_x = pretrained_embedding(x)     
            
        h0 = torch.zeros((self.config.num_layers,self.config.batch_size,self.config.hidden_size), device=device)
        c0 = torch.zeros((self.config.num_layers,self.config.batch_size,self.config.hidden_size), device=device)        
        output, (ht, ct) = self.lstm(embed_x,(h0, c0))      
        output = self.tanh(output)
        output  = output .permute(0,2,1)
        ht = self.maxpool(output)
        ht = torch.squeeze(ht)
        out = F.dropout(input=ht, p=0.5)
        out = self.linear(ht)
        out = torch.squeeze(out)
        out  = self.sigmoid(out)
        #print('out size 4 ',out.size())
        return out
    
        
class LSTMconfiguration():    
    def __init__(self,vocab_size, embedding_size, hidden_size, num_layers, num_class, sentence_length, maxpool_kernel, batch_size,pre_embed = False):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_class = num_class
        self.sentence_length = sentence_length
        self.maxpool_kernel = maxpool_kernel
        self.batch_size = batch_size
        self.pre_embed = pre_embed