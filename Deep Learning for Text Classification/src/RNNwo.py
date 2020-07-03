'''
Here is training process for RNN w/o pretrained embed
'''

import torch
import torch.nn as nn
from util import get_traindataloader,get_testdataloader,mkdir
import time
from module import TextRNN,LSTMconfiguration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def RNNwo():
    mkdir("result")
    mkdir("model")
    ftest_loss = open("result/LSTMwo_test_loss.txt","w")
    ftest_acc = open("result/LSTMwo_test_accuracy.txt","w")
    ftrain_loss = open("result/LSTMwo_train_loss.txt","w")
    ftrain_acc = open("result/LSTMwo_train_accuracy.txt","w")
    
            
    def evaluate(layer,model_path,epoch,flag = "test"):
           
        layer.load_state_dict(torch.load(model_path))
        criterion = nn.BCELoss()
        
        accuracy = 0
        total_loss = 0
        
        if flag == "test":
            with torch.no_grad():
                for test_input, test_label in test_generator:
                    predict = layer(test_input)
                    predict = predict.round()
                    
                    a = (predict == test_label)
                    accuracy = accuracy + a.float().sum()/(len(test_generator)*len(a))
                    
                    loss = criterion(predict, test_label)
                    total_loss = total_loss + float(loss)
                    
                print(f"epoch: {epoch}  test_total_loss: {total_loss}")
                print(f"{epoch} {total_loss}",file = ftest_loss)
                
                print(f"epoch: {epoch}  test_accuracy: {accuracy}")
                print(f"{epoch} {accuracy}",file = ftest_acc)
    
                 
        elif flag == "train":
            
            with torch.no_grad():
                for train_input, train_label in train_generator:
                    predict = layer(train_input)
                    predict = predict.round()
                    
                    a = (predict == train_label)
                    accuracy = accuracy + a.float().sum()/(len(test_generator)*len(a))
                    
                    loss = criterion(predict, train_label)
                    total_loss = total_loss + float(loss)
                    
                print(f"epoch: {epoch}  train_total_loss: {total_loss}")
                print(f"{epoch}: {total_loss}",file = ftrain_loss) 
                    
                print(f"epoch: {epoch}  train_accuracy: {accuracy}")
                print(f"{epoch} {accuracy}",file = ftrain_acc)
        else:
            print("flag wrong format")   
            
    config =  LSTMconfiguration(vocab_size= 10000,embedding_size = 100, hidden_size = 100, num_layers = 1, num_class = 1,sentence_length = 300, maxpool_kernel = 4,batch_size = 10,pre_embed = False)   
    batch_size = config.batch_size
    train_generator = get_traindataloader(batch_size)
    test_generator = get_testdataloader(batch_size)
    
    layer = TextRNN(config)    
    layer = layer.to(device)
    criterion = nn.BCELoss()
     
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
    batch_size = config.batch_size
    train_generator = get_traindataloader(batch_size)
    test_generator = get_testdataloader(batch_size)
    max_epochs = 50
    
    model_path = "model/LSTMwo.pkl"
        
    start_t = time.perf_counter()
    for epoch in range(max_epochs):
        print(f"epoch: {epoch+1}")
        for batch_input, batch_labels in train_generator:
            optimizer.zero_grad()
            predict = layer(batch_input)
            loss = criterion(predict, batch_labels)
            loss.backward()
            optimizer.step()
    
        torch.save(layer.state_dict(), model_path)
        
        evaluate(layer,model_path,epoch+1,flag = "test")
        evaluate(layer,model_path,epoch+1,flag = "train")
        
    end_t = time.perf_counter()
    print("LSTMwo time %.3f:"%(end_t-start_t))
    
    end_t = time.perf_counter()
    
    
    
    ftest_acc.close()
    ftest_loss.close()      
    ftrain_acc.close()
    ftrain_loss.close()
