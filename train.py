import config
from dataset import My_dataloader
import datapreprocess
import engine
from model import My_model
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

df = datapreprocess.csv_process()
#df = df.drop(['labels'])
#print(df.head())
d_train, d_test = train_test_split(df, test_size=0.2)

# Now we will create a vocabulary dictionary for training data

wordtoid = {'<PAD>':0,'<UNK>':1}
count = 2

for i,row in d_train.iterrows():

    tokens = row['data'].lower().split()

    for token in tokens:

        if token not in wordtoid:

            wordtoid[token] = count
            count += 1

vocab = len(wordtoid)

idtoword = {v:k for k,v in wordtoid.items()}

# Now we will convert all the training and testing sentences into token numbers

train_input = []
train_targets = []
for i,row in d_train.iterrows():

    line_input = []

    tokens = row['data'].lower().split()
    for token in tokens:
        line_input.append(wordtoid[token])

    train_input.append(line_input)
    train_targets.append(row['targets'])



#print(train_input[100])

#y = max((x) for x in train_input)

#print(y)
test_targets = []
test_input = []
for i,row in d_test.iterrows():

    tokens = row['data'].lower().split()
    line_input = []
    for token in tokens:

        if token not in wordtoid:

            line_input.append(wordtoid['<UNK>'])

        else:

            line_input.append(wordtoid[token])

    test_input.append(line_input)
    test_targets.append(row['targets'])

max_train_len = max(len(x) for x in train_input)
max_test_len = max(len(x) for x in  test_input)
#print(max_train_len)

# Now we will pad zeros at the starting of the input sentence
if max_train_len > max_test_len:

    pad_size = max_train_len

else:
    pad_size = max_test_len

for i in range(len(train_input)):
    pad = [wordtoid['<PAD>']] * (pad_size - len(train_input[i]))
    train_input[i] = pad + train_input[i]
# Padding in test input



for i in range(len(test_input)):
    pad = [wordtoid['<PAD>']]*(pad_size - len(test_input[i]))
    test_input[i] = pad + test_input[i]
#print(test_input[20])

train_input = np.array(train_input)
train_targets = np.array(train_targets)
test_input = np.array(test_input)
test_targets = np.array(test_targets)

train_input = torch.from_numpy(train_input.astype(np.float32))
train_targets = torch.from_numpy(train_targets.astype(np.float32))
test_input = torch.from_numpy(test_input.astype(np.float32))
test_targets = torch.from_numpy(test_targets.astype(np.float32))



#print(train_input.shape)
#print(test_input.shape)

train_dataloader = My_dataloader(train_input,train_targets)
val_dataloader = My_dataloader(test_input,test_targets)
model = My_model(embd_dim=8,vocab=vocab, hiddenSize= 10, numLayers= 4)
optimizer = torch.optim.Adam(model.parameters())
train_load = torch.utils.data.DataLoader(
    train_dataloader, 
    batch_size = config.TRAIN_BATCH_SIZE,
    shuffle = True)

test_load = torch.utils.data.DataLoader(
    val_dataloader,
    batch_size = config.VAL_BATCH_SIZE )



for i in range(config.EPOCHS):

    loss, acc = engine.train_one_step(
        train_dataloader= train_load, 
        model=model,
        optimizer= optimizer,
        device='cuda:0')
    
    print(f"Epoch {i+1} : Training loss {loss}, accuracy {acc}")


val_loss,val_acc = engine.val_one_step(
    val_dataloader=test_load,
    model = model,
    device= 'cuda:0'
)

print(f"Testing Loss is {val_loss}, Testing Accuracy is {val_acc}")
