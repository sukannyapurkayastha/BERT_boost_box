import torch


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


loss_function = torch.nn.CrossEntropyLoss()


def train(model, epoch, training_loader, optimizer, device):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    print(f'No of correct examples before training:{n_correct}')
    print(f'No of examples before training:{nb_tr_examples}')
    for _, data in enumerate(training_loader, 0):
        ids = data['ids']
        mask = data['mask']
        targets = data['targets']
        token_type_ids = data['token_type_ids']
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        # predictions = torch.nn.Softmax(outputs)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
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

    print(f'No of correct examples after training:{n_correct}')
    print(f'No of examples after training:{nb_tr_examples}')
    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    return


def valid(model, epochs, testing_loader, device, type_of_data):
    model.eval()
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    pred_idx = []

    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids']
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # if _ % 5000 == 0:
            #    loss_step = tr_loss / nb_tr_steps
            #    accu_step = (n_correct * 100) / nb_tr_examples
            #    print(f"Validation Loss per 100 steps: {loss_step}")
            #    print(f"Validation Accuracy per 100 steps: {accu_step}")
            pred_idx.extend(big_idx.tolist())
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Test Loss: {epoch_loss}")
    print(f"Test Accuracy : {epoch_accu}")

    if type_of_data == 'test':
        f = open(f'../Results/Result_{epochs}.txt', 'w')
        for idx in pred_idx:
            f.write(str(idx) + '\n')
        f.close()

    return epoch_accu, epoch_loss
