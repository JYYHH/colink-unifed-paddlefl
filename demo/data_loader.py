import json
import logging
import os

import numpy as np
import pickle

VOCAB_SIZE = 0
my_vocab = {}

def word_index_train(word):
    global VOCAB_SIZE
    global my_vocab

    if word in my_vocab:
        return my_vocab[word]
    else:
        my_vocab[word] = VOCAB_SIZE
        VOCAB_SIZE += 1
        return VOCAB_SIZE - 1

def word_index_test(word):
    global VOCAB_SIZE
    global my_vocab

    if word in my_vocab:
        return my_vocab[word]
    else:
        return 3

def trans_train(x):
    return np.array([[word_index_train(word) for word in x_item] for x_item in x], dtype = np.int64)

def trans_test(x):
    return np.array([[word_index_test(word) for word in x_item] for x_item in x], dtype = np.int64)


def read_data(dataset_switch, train_data_dir, test_data_dir):
    if dataset_switch == 0:
        train_data,test_data = [],[]
        
        train_files = os.listdir(train_data_dir)
        train_files = [f for f in train_files if f.endswith('.json') and f[0]!='_']
        for f in train_files:
            file_path = os.path.join(train_data_dir, f)
            my_data = np.array(json.load(open(file_path, 'r'))["records"])
            train_data.append(my_data)

        
        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.json') and f[0]!='_']
        for f in test_files:
            file_path = os.path.join(test_data_dir, f)
            my_data = np.array(json.load(open(file_path, 'r'))["records"])
            test_data.append(my_data)

        return train_data, test_data
    else:
        global VOCAB_SIZE
        VOCAB_SIZE = 4
        global my_vocab
        my_vocab = {'<PAD>' : 0, '<BOS>' : 1, '<EOS>' : 2, '<OOV>' : 3}

        train_data, test_data = [], []
        
        train_files = os.listdir(train_data_dir)
        train_files = [f for f in train_files if f.endswith('.json') and f[0]!='_']
        for f in train_files:
            nx, ny = np.array([], dtype = np.int64).reshape(-1, 10), np.array([], dtype = np.int64).reshape(-1, 10)

            file_path = os.path.join(train_data_dir, f)
            my_data = json.load(open(file_path, 'r'))["records"]

            for x in my_data:
                ny = np.vstack((ny, trans_train(x[0]['target_tokens'])))
                nx = np.vstack((nx, trans_train(x[1])))
                # print(client_name, train_data[client_name]['y'].shape, train_data[client_name]['x'].shape)

            
            train_data.append(np.stack((nx, ny), axis = 0).transpose(1, 0, 2))
            # print(np.stack((nx, ny), axis = 0).transpose(1, 0, 2).shape)


        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.json') and f[0]!='_']
        for f in test_files:
            nx, ny = np.array([], dtype = np.int64).reshape(-1, 10), np.array([], dtype = np.int64).reshape(-1, 10)

            file_path = os.path.join(test_data_dir, f)
            my_data = json.load(open(file_path, 'r'))["records"]

            for x in my_data:
                ny = np.vstack((ny, trans_test(x[0]['target_tokens'])))
                nx = np.vstack((nx, trans_test(x[1])))
                # print(client_name, train_data[client_name]['y'].shape, train_data[client_name]['x'].shape)

            test_data.append(np.stack((nx, ny), axis = 0).transpose(1, 0, 2))

        pickle.dump(VOCAB_SIZE, open("vocab.pkl", "wb"))
        return train_data, test_data