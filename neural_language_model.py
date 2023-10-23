import torch
import torch.nn as nn
from tqdm import tqdm
import math
from tokenization import tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import sys

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
pad_index = 0
m = 0
class datasets():
    def __init__(self,file_name) -> None:
        self.vocab = {}
        self.contexts = []
        self.maxi = 0
        self.batch_size = 256
        self.file_name = file_name
    def preprocess(self,text):
    
        tk = tokenizer()
        txt = tk.substitute(text)
        txt = tk.Punctuations(txt)
        sents = txt.split(" . ")
        
        vocab_freq = {}
        for line in sents:
            token = line.split()
            self.maxi = max(self.maxi,len(token))
            for word in token:
                if word in vocab_freq:
                    vocab_freq[word] += 1
                else:
                    vocab_freq[word] = 1
            try:
                pref = list()
                pref = [token[0]]
                for t in list(token[1:]):
                    pref.append(t)
                    copied = pref.copy()
                    self.contexts.append(copied)
            except:
                continue
        vocab = {}
        for items,val in vocab_freq.items():
            if val < 2:
                if "<UNK>" in vocab:
                    vocab["<UNK>"] += val
                else:
                    vocab["<UNK>"] = 1
            else:
                vocab[items] = val
        
        self.vocab = vocab

        self.words_to_indices = {word: index for index, word in enumerate(self.vocab)}

        pad_index = len(self.vocab) - 1
        for i in range(len(self.contexts)):
            seq = []
            for w in self.contexts[i]:
                try: seq.append(self.words_to_indices[w])
                except: seq.append(len(self.vocab) - 2)
            curr_seq = seq
        # padding
        self.contexts[i] = [pad_index]*(self.maxi-len(curr_seq)) + curr_seq
        # 2D tensor
        self.batches = torch.split(torch.tensor(self.contexts), self.batch_size)
        # print(self.batches)
        # store the prefix , last batches to make the model to learn
        self.learn_batches = list()
        for batch in self.batches:
            self.learn_batches.append((batch[:,:-1], batch[:,-1]))

    def load_file(self):
        with open(self.file_name, 'r') as f: 
            text = f.read()
        self.preprocess(text)
# datasets().preprocess()

class LSTM_Model(nn.Module):
    def __init__(self,embed_dim: int,hidd_dim: int,learning_rate,epsi,trainset) -> None:
        super().__init__()
        self._embed_dim_ = embed_dim
        self._hid_dim = hidd_dim
        self.learning_rate = learning_rate
        self.epsi = epsi
        vocab = trainset.vocab
        vocab_size = len(vocab)
        self.embedding = nn.Embedding(vocab_size,
                                    embed_dim)
        self.lstm = nn.LSTM(embed_dim,
                            hidd_dim)
        self.res = nn.Linear(hidd_dim,
                            vocab_size)
    def forward(self, hist):
        embed_output = self.embedding(hist)
        lstm_ouput, hid = self.lstm(embed_output)
        result = lstm_ouput[:,-1] # Last Layer is taken
        result = self.res(result)

        return result

    def initialize_hidden(self):
        pass

class Train():
    def __init__(self,model):
        self.model = model
        self.p_loss = -math.inf
    def train(self,dataset):
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.model.parameters(),self.model.learning_rate)
        batches_learn = dataset.learn_batches
        print(len(batches_learn))
        for i_epoch in range(0,10):
            avg_loss = 0
            for i, (hist,words) in enumerate(batches_learn):
                optimiser.zero_grad()
                pred = self.model.forward(hist)
                loss = criterion(pred,words)
                avg_loss += loss.item()
                loss.backward()
                optimiser.step()
                avg_loss = avg_loss/len(batches_learn)
        
            print(i_epoch,avg_loss)
            
            if(0 <= self.model.epsi-abs(avg_loss - self.p_loss)): 
                break
            self.p_loss = avg_loss

    

TRAINSET = datasets('./pre.txt')
TESTSET = datasets('')
TRAINSET.load_file()
saved_model = './model_2.pth'
loaded_model = LSTM_Model(100, 150, 0.001, 0.001, TRAINSET)
loaded_model.load_state_dict(torch.load(saved_model))

print(get_per(loaded_model,TRAINSET))

  
    