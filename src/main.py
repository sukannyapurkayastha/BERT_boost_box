# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import config
import dataset
import torch
import h5py
import pandas as pd
import model_class
from train_test_eval import train, valid
from torch.utils.data import DataLoader
from torch import cuda
import torch.nn as nn 
import argparse


device = 'cuda' if cuda.is_available() else 'cpu'

def run(train_dataset, test_dataset, epochs, alpha, lr):
    df_train = pd.read_csv(train_dataset, sep='\t', names=['text', 'relation', 'relation_label'])
    df_test = pd.read_csv(test_dataset, sep='\t', names=['text', 'relation', 'relation_label'])

    with h5py.File('../data/train_mask.h5', 'r') as hf:
        data_train = hf['train_mask'][:].tolist()
    # with h5py.File('../data/test_mask.h5', 'r') as hf:
     #   data_test = hf['test_mask'][:].tolist()

    train_data_set = dataset.BERT_KBQA_Dataloader(df_train.text.values, df_train.relation_label.values, data_train)
    #test_data_set = dataset.BERT_KBQA_Dataloader(df_test.text.values, df_train.relation_label.values, data_test)

    train_data_loader = DataLoader(train_data_set, batch_size=config.TRAIN_BATCH_SIZE)
    # test_data_loader = DataLoader(test_data_set, batch_size=config.VALID_BATCH_SIZE)

    model = model_class.Bert_Kbqa_Model()
    model.to(device)
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        train(model, epoch, alpha, train_data_loader, optimizer)
        #valid(model,valid_data_loader)

    del data_train
    del train_data_loader
    del train_data_set

    with h5py.File('../data/test_mask.h5','r') as hf:
        data_test = hf['test_mask'][:].tolist()
    test_data_set = dataset.BERT_KBQA_Dataloader(df_test.text.values, df_test.relation_label.values, data_test)
    test_data_loader = DataLoader(test_data_set, batch_size=config.VALID_BATCH_SIZE,)
    acc = valid(model, test_data_loader)
    print("Accuracy on test data = %0.2f%%" % acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help='No. of epochs', type=int)
    parser.add_argument("--alpha", help='Value of alpha', type=float)
    parser.add_argument("--lr", help='learning_rate', type=float)
    train_dataset = '../data/train_data_final.txt'
    test_dataset = '../data/test_data_final.txt'
    run(train_dataset, test_dataset, args.epochs, args.alpha, args.lr)


