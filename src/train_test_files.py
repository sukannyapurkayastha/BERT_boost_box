import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
#optimizer = torch.optim.Adam(lr=3e-5)
alpha = 0.5

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


loss_function = torch.nn.CrossEntropyLoss()


def mask_loss(outputs, target_mask):
    return torch.nn.BCEWithLogitsLoss()(outputs, target_mask)


def train(model, epoch, training_loader, optimizer):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids']
        mask = data['mask']
        targets = data['targets']
        mask_labels = data['mask_labels']
        token_type_ids = data['token_type_ids']
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        mask_labels = mask_labels.to(device, dtype=torch.float)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        mask_outputs = outputs.type_as(mask_labels)
        # predictions = torch.nn.Softmax(outputs)
        Loss_CE = loss_function(outputs, targets)
        Loss_Mask = mask_loss(mask_outputs, mask_labels)
        loss = Loss_CE + alpha * Loss_Mask
        tr_loss += Loss_CE.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return


def valid(model, testing_loader):
    model.eval()
    n_correct = 0
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    n_wrong = 0
    total = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask).squeeze()
            mask_labels = data['mask_labels'].to(device, dtype=torch.long)
            outputs_ = outputs * mask_labels
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu
