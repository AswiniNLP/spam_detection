import config
import torch
from tqdm import tqdm


def train_one_step(train_dataloader, model, optimizer, device):

    final_loss = 0
    final_accuracy = 0

    for data in tqdm(train_dataloader):

        # Move the data and model to GPU

        optimizer.zero_grad()

        for k,v in data.items():
            data[k] = v.to(device)

        model = model.to(device)
        pred,loss = model(**data)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy = (pred==data['targets']).float().mean()
            #print(pred.shape)
            #print(data['targets'].shape)
        final_loss += loss
        final_accuracy += accuracy
    

    return final_loss/len(train_dataloader), final_accuracy/len(train_dataloader)


def val_one_step(val_dataloader, model, device):

    final_loss = 0 
    final_accuracy = 0

    for data in tqdm(val_dataloader):

        for k,v in data.items():

            data[k] = v.to(device)

        model = model.to(device)

        pred,loss = model(**data)

        with torch.no_grad():
            accuracy = (pred==data['targets']).float().mean()

        final_loss += loss
        final_accuracy += accuracy

    return final_loss/len(val_dataloader), final_accuracy/len(val_dataloader)