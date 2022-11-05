import csv 
import torch
import torch.utils.data
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.optim import Adam
from torch.autograd import Variable
from RecurrentNet import RecurrentNet



def testAccuracy(model):
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            audio, labels = data
            # run the model on the test set to predict labels
            outputs = model(audio)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    accuracy = (100 * accuracy / total)
    return(accuracy)



def train(net, train_loader, optimizer):
    total=0
    correct=0
    for batch_id, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()    # zero the gradients
        output = net(data)       # apply network
        loss = F.binary_cross_entropy(output,target)
        loss.backward()          # compute gradients
        optimizer.step()         # update weights
        pred = (output >= 0.5).float()
        correct += (pred == target).float().sum()
        total += target.size()[0]
        accuracy = 100*correct/total

    if epoch % 100 == 0:
        print('ep:%5d loss: %6.4f acc: %5.2f' %
             (epoch,loss.item(),accuracy))

    return accuracy     
  
parser = argparse.ArgumentParser()
parser.add_argument('--layers',type=int)
parser.add_argument('--seqsize', type=int)
parser.add_argument('--dropout', type=float,default=0)
parser.add_argument('--netType', type=str)
parser.add_argument('--epoch', type=int,default='100', help='max training epochs')
args = parser.parse_args()
data_path = "D:\\MLdata\\features.csv"
#df = pd.read_csv('features.csv')
data = pd.read_csv(data_path, index_col=0, header=[0, 1, 2])

num_input = data.shape[1] - 1

full_input  = data[:,0:num_input]
full_target = data[:,num_input:num_input+1]

train_dataset = torch.utils.data.TensorDataset(full_input,full_target)
train_loader  = torch.utils.data.DataLoader(train_dataset,
                                 batch_size=train_dataset.__len__())

net = ReccurentNet(n_layers = args.layers, seq_size = args.seqsize, dropout = args.dropout, netType = args.netType)

optimizer = torch.optim.Adam(net.parameters(),lr=args.lr,weight_decay=0.00001)

    # training loop
epoch = 0
count = 0
while epoch < args.epoch and count < 2000:
    epoch = epoch+1
    accuracy = train(net, train_loader, optimizer)
    if accuracy == 100:
        count = count+1
    else:
        count = 0