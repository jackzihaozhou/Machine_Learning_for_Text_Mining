# Deep Learning for Text Classification

The src structure is as follow:

HW3.zip
├── README.txt
└── src
	├── data
    ├── CNN.py
    ├── LSTM.py
    ├── CNNwo.py
    ├── CNNw.py
    ├── data(should be replaced to real 'data' files extracted from 'data.tar.gz')
    ├── module.py
    ├── pretrain_weight.py
    ├── RNNwo.py
    ├── RNNw.py
    └── util.py

1.Running RNN w/o pretrained embedding, 	
	```bash
		python3 LSTM.py wo
	```
	this will create folders(if existed do noting) 'result' and 'model' to:
	1 .save the info of every epoch during training(accuracy & loss), i.e., the following four files:
		"result/RNNwo_test_loss.txt","w")
		"result/RNNwo_test_accuracy.txt","w")
		"result/RNNwo_train_loss.txt","w")
		"result/RNNwo_train_accuracy.txt","w")
	2. save the model every update, i.e., 
		"model/LSTMwo.pkl"
	
	The following three are the same will create files like this.
	
1.Running RNN w pretrained embedding, 	
	```bash
		python3 LSTM.py w
	```

1.Running CNN w/o pretrained embedding, 	
	```bash
		python3 CNN.py wo
	```

1.Running CNN w pretrained embedding, 	
	```bash
		python3 CNN.py w
	```