import os
import sys
import pandas as pd
import numpy as np
import json
from data_loader import * 


config = json.load(open('config.json', 'r'))

dataset_name = (
                'femnist',
                'reddit'
                )

Train_path = [
              '../data/femnist/train/',
              '../data/reddit/train/',
                ]
Test_path = [
              '../data/femnist/test/',
              '../data/reddit/test/',
                ]

try:     
    dataset_switch = dataset_name.index(config["dataset"])
    train_data, test_data = read_data(dataset_switch, Train_path[dataset_switch], Test_path[dataset_switch])
    permutation = np.arange(len(train_data))
    ret_num = 0
except:
    pass

def transfer(x):
    if dataset_switch == 0 and config["model"] == 'lenet':
        x = np.pad(
                x.reshape(28, 28), 
                ((2, 2), (2, 2)),
                'reflect'
            ).reshape(1, 32, 32)
    return x

def CreateReader(test = False, ID = 0):
    def train_reader():
        my_csv = train_data[ID]
        for i in range(my_csv.shape[0]):
            yield list(transfer(my_csv[i][1:])) , int(my_csv[i][0])
    def test_reader():
        for my_csv in test_data:
            for i in range(my_csv.shape[0]):
                yield list(transfer(my_csv[i][1:])) , int(my_csv[i][0])   

    if test:
        return test_reader
    else:
        return train_reader

def CreateReaderReddit(test = False, ID = 0):
    def train_reader():
        my_csv = train_data[ID]
        for i in range(my_csv.shape[0]):
            yield np.array(my_csv[i][0]).reshape(-1, 1) , np.array(my_csv[i][1]).reshape(-1, 1)
    def test_reader():
        for my_csv in test_data:
            for i in range(my_csv.shape[0]):
                yield np.array(my_csv[i][0]).reshape(-1, 1) , np.array(my_csv[i][1]).reshape(-1, 1) 

    if test:
        return test_reader
    else:
        return train_reader
    
def LEAFTrain(ID):
    global ret_num
    ret_num += 1
    if (ret_num - 1) % config["training_param"]["client_per_round"] == 0:
        np.random.shuffle(permutation)

    if dataset_switch == 0:
        return CreateReader(False, permutation[ID]) 
    else:
        return CreateReaderReddit(False, permutation[ID]) 
def LEAFTest():
    if dataset_switch == 0:
        return CreateReader(True)
    else:
        return CreateReaderReddit(True)