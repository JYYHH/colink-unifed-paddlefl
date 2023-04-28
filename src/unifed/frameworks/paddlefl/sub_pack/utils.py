import os
import sys
import pandas as pd
import numpy as np
import json

config = json.load(open('config.json', 'r'))

dataset_name = ('breast_horizontal',
                'default_credit_horizontal',
                'give_credit_horizontal',
                'student_horizontal',
                'vehicle_scale_horizontal',
                'femnist',
                'reddit'
                )

Train_path = ['../csv_data/breast_horizontal_train/breast_homo_',
              '../csv_data/default_credit_horizontal_train/default_credit_homo_',
              '../csv_data/give_credit_horizontal_train/give_credit_homo_',
              '../csv_data/student_horizontal_train/student_homo_',
              '../csv_data/vehicle_scale_horizontal_train/vehicle_scale_homo_',
              '../data/femnist/train/',
              '../data/reddit/train/'
                ]
Test_path = ['../csv_data/breast_horizontal_test/breast_homo_',
             '../csv_data/default_credit_horizontal_test/default_credit_homo_',
             '../csv_data/give_credit_horizontal_test/give_credit_homo_',
             '../csv_data/student_horizontal_test/student_homo_',   
             '',
             '../data/femnist/test/',
             '../data/reddit/test/'
                ]

dataset_switch = dataset_name.index(config["dataset"])
train_path = (Train_path[dataset_switch] + 'host' + '_1' * int(dataset_switch==1) + '.csv', Train_path[dataset_switch] + 'guest.csv')
if dataset_switch < 4:
    test_path = Test_path[dataset_switch] + 'test.csv'
elif dataset_switch == 4:
    test_path = train_path
else:
    pass

def CreateReader(filename):
    def reader():
        my_csv = np.array(pd.read_csv(filename))
        for i in range(my_csv.shape[0]):
            yield list(my_csv[i][2:]) , int(my_csv[i][1])
    def double_reader():
        my_csv = np.array(pd.read_csv(filename[0]))
        for i in range(my_csv.shape[0]):
            yield list(my_csv[i][2:]) , int(my_csv[i][1])     
        my_csv = np.array(pd.read_csv(filename[1]))
        for i in range(my_csv.shape[0]):
            yield list(my_csv[i][2:]) , int(my_csv[i][1])            
    if isinstance(filename, str):
        return reader
    return double_reader
    
def Train(ID):
    return CreateReader(train_path[ID]) 
def Test():
    return CreateReader(test_path)

def Batch(reader,batch_size,drop_last = False):
    def batch_reader():
        r = reader()
        b = []
        for instance in r:
            b.append(instance)
            if len(b) == batch_size:
                yield b
                b = []
        if drop_last is False and len(b) != 0:
            yield b

    return batch_reader
