import torch
from torch import cuda
#device = 'cuda' if cuda.is_available() else 'cpu'
#optimizer = torch.optim.Adam(lr=3e-5)
#alpha = 0.5

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


loss_function = torch.nn.CrossEntropyLoss()


def mask_loss(outputs, target_mask):
    return torch.nn.BCEWithLogitsLoss()(outputs, target_mask)


def train(model, epoch, alpha, training_loader, optimizer, device):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    print(f'No of correct examples before training:{n_correct}')
    print(f'No of examples after training:{nb_tr_examples}')
    for _, data in enumerate(training_loader, 0):
        ids = data['ids']
        mask = data['mask']
        targets1 = data['targets1']
        targets2 = data['targets2']
        targets3 = data['targets3']
        targets4 = data['targets4']
        targets5 = data['targets5']
        mask_labels = data['mask_labels']
        token_type_ids = data['token_type_ids']
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets1 = targets1.to(device, dtype=torch.long)
        targets2 = targets2.to(device, dtype=torch.long)
        targets3 = targets3.to(device, dtype=torch.long)
        targets4 = targets4.to(device, dtype=torch.long)
        targets5 = targets5.to(device, dtype=torch.long)
        mask_labels = mask_labels.to(device, dtype=torch.float)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        mask_outputs = outputs.type_as(mask_labels)
        # predictions = torch.nn.Softmax(outputs)
        Loss_1 = loss_function(outputs, targets1)
        Loss_2 = loss_function(outputs, targets2)
        Loss_3 = loss_function(outputs, targets3)
        Loss_4 = loss_function(outputs, targets4)
        Loss_5 = loss_function(outputs, targets5)
        Loss_Mask = mask_loss(mask_outputs, mask_labels)
        loss = (1-alpha) * (Loss_1-Loss_2-Loss_3-Loss_4-Loss_5) + alpha * Loss_Mask
        tr_loss += Loss_1.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets1)

        nb_tr_steps += 1
        nb_tr_examples += targets1.size(0)

        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'No of correct examples after training:{n_correct}')
    print(f'No of examples after training:{nb_tr_examples}')
    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    return


def valid(model, epochs, testing_loader, device, alpha, type_of_data):
    model.eval()
    n_correct = 0
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    n_wrong = 0
    total = 0
    pred_idx=[]
    topk_idx_list=[]
        
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids']
            mask_labels = data['mask_labels'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            #outputs = outputs * mask_labels
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            val_topk, idxs_topk = torch.topk(outputs.data, 5)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # if _ % 5000 == 0:
            #    loss_step = tr_loss / nb_tr_steps
            #    accu_step = (n_correct * 100) / nb_tr_examples
            #    print(f"Validation Loss per 100 steps: {loss_step}")
            #    print(f"Validation Accuracy per 100 steps: {accu_step}")
            pred_idx.extend(big_idx.tolist())
            topk_idx_list.extend(idxs_topk.tolist())
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Test Loss: {epoch_loss}")
    print(f"Test Accuracy : {epoch_accu}")
    
    if type_of_data=='test':
        f=open(f'../Results_hard_neg/Result_{alpha}_{epochs}.txt','w')
        for idx in pred_idx:
            f.write(str(idx)+'\n')
        f.close()
    elif type_of_data=='train':
        f=open(f'../Results_hard_neg/Result_{alpha}_{epochs}_train.txt','w')
        for idx in topk_idx_list:
            f.write(str(idx)+'\n')
        f.close()

    return epoch_accu, epoch_loss
