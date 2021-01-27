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
from pytorchtools import EarlyStopping


device = 'cuda' if cuda.is_available() else 'cpu'

def run(train_dataset, valid_dataset, test_dataset, epochs, alpha, lr, batch_size):
    df_train = pd.read_csv(train_dataset, sep='\t', names=['text', 'relation', 'relation_label'])
    df_valid = pd.read_csv(valid_dataset, sep='\t', names=['text', 'relation', 'relation_label'])
    df_test = pd.read_csv(test_dataset, sep='\t', names=['text', 'relation', 'relation_label'])

    with h5py.File('../data/train_mask.h5', 'r') as hf:
        mask_train = hf['train_mask'][:].tolist()
    with h5py.File('../data/valid_mask.h5', 'r') as hf:
        mask_valid = hf['valid_mask'][:].tolist()
    # with h5py.File('../data/test_mask.h5', 'r') as hf:
     #   data_test = hf['test_mask'][:].tolist()

    train_data_set = dataset.BERT_KBQA_Dataloader(df_train.text.values, df_train.relation_label.values, mask_train)
    valid_data_set = dataset.BERT_KBQA_Dataloader(df_valid.text.values, df_valid.relation_label.values, mask_valid)
    #test_data_set = dataset.BERT_KBQA_Dataloader(df_test.text.values, df_train.relation_label.values, data_test)

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data_set, batch_size = batch_size)
    # test_data_loader = DataLoader(test_data_set, batch_size=config.VALID_BATCH_SIZE)

    model = model_class.Bert_Kbqa_Model()
    model.to(device)
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=f'../checkpoints/checkpoint_{alpha}_{batch_size}_{epochs}.pt')

    for epoch in range(epochs):
        train(model, epoch, alpha, train_data_loader, optimizer, device)
        _, loss = valid(model,valid_data_loader, device, alpha, 'valid')
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(f'../checkpoints/checkpoint_{alpha}_{batch_size}_{epochs}.pt'))
    f=open(f'../checkpoints/checkpoint_{alpha}_{batch_size}_{epochs}.txt','w')
    f.write(f'Epochs:{epochs}\n Batch size:{batch_size}\n Learning Rate:{lr}\n alpha: {alpha}')
    f.close()
    #torch.save(model.state_dict(), f'../checkpoints/checkpoint_{alpha}.pt')

    del mask_train
    del mask_valid
    del train_data_loader
    del train_data_set
    del valid_data_loader
    del valid_data_set

    with h5py.File('../data/test_mask.h5','r') as hf:
        data_test = hf['test_mask'][:].tolist()
    test_data_set = dataset.BERT_KBQA_Dataloader(df_test.text.values, df_test.relation_label.values, data_test)
    test_data_loader = DataLoader(test_data_set, batch_size=batch_size)
    acc, _ = valid(model, test_data_loader, device, alpha, 'test')
    print("Accuracy on test data = %0.2f%%" % acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help='No. of epochs', type=int)
    parser.add_argument("--alpha", help='Value of alpha', type=float)
    parser.add_argument("--lr", help='learning_rate', type=float)
    parser.add_argument("--batch_size", help='batch_size', type=int)
    args = parser.parse_args()
    train_dataset = '../data/train_data_final.txt'
    valid_dataset = '../data/valid_data_final.txt'
    test_dataset = '../data/test_data_final.txt'
    run(train_dataset, valid_dataset, test_dataset, args.epochs, args.alpha, args.lr, args.batch_size)


