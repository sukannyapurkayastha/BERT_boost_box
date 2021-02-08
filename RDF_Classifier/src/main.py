# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import dataset
import torch
import os
import pandas as pd
import model_class
from train_test_eval import train, valid
from torch.utils.data import DataLoader
from torch import cuda
import torch.nn as nn
import argparse
from pytorchtools import EarlyStopping


device = 'cuda' if cuda.is_available() else 'cpu'


def run(train_dataset, valid_dataset, test_dataset, epochs, lr, batch_size, patience):
    df_train = pd.read_csv(train_dataset, sep='\t', names=['text', 'relation_label'])
    df_valid = pd.read_csv(valid_dataset, sep='\t', names=['text', 'relation_label'])
    df_test = pd.read_csv(test_dataset, sep='\t', names=['text', 'relation_label'])

    train_data_set = dataset.RDF_DataLoader(df_train.text.values, df_train.relation_label.values)
    valid_data_set = dataset.RDF_DataLoader(df_valid.text.values, df_valid.relation_label.values)

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data_set, batch_size=batch_size)
    # test_data_loader = DataLoader(test_data_set, batch_size=config.VALID_BATCH_SIZE)

    model = model_class.RDF_classifier_Model()
    model.to(device)
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    path = f'../checkpoints/checkpoint_{batch_size}_{epochs}_{patience}.pt'
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    for epoch in range(epochs):
        train(model, epoch, train_data_loader, optimizer, device)
        _, valid_loss = valid(model, epochs, valid_data_loader, device, 'valid')
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(f'../checkpoints/checkpoint_{batch_size}_{epochs}_{patience}.pt'))
    f = open(f'../checkpoints/checkpoint_{batch_size}_{epochs}_{patience}.txt', 'w')
    f.write(f'Epochs:{epochs}\n Batch size:{batch_size}\n Learning Rate:{lr}\n patience: {patience}\n')
    f.close()

    del train_data_loader
    del train_data_set
    del valid_data_loader
    del valid_data_set

    test_data_set = dataset.RDF_DataLoader(df_test.text.values, df_test.relation_label.values)
    test_data_loader = DataLoader(test_data_set, batch_size=batch_size)
    acc, _ = valid(model, epochs, test_data_loader, device, 'test')
    print("Accuracy on test data = %0.2f%%" % acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help='Path to Dataset', type=str)
    parser.add_argument("--epochs", help='No. of epochs', type=int)
    parser.add_argument("--lr", help='learning_rate', type=float)
    parser.add_argument("--batch_size", help='batch_size', type=int)
    parser.add_argument("--patience", help='patience', type=int)
    args = parser.parse_args()
    train_dataset = os.path.join(args.path, 'train_rdf_data.txt')
    valid_dataset = os.path.join(args.path, 'valid_rdf_data.txt')
    test_dataset = os.path.join(args.path, 'test_rdf_data.txt')
    run(train_dataset, valid_dataset, test_dataset, args.epochs, args.lr, args.batch_size, args.patience)
