import config
import torch
import torch.nn as nn

def loss_fn(output, targets):

    return torch.nn.CrossEntropyLoss()(output, targets)

class My_model(nn.Module):

    def __init__(self, embd_dim, vocab, hiddenSize, numLayers):

        super(My_model, self).__init__()
        #self.input_size = inputSize
        self.hidden_size = hiddenSize
        self.num_layers = numLayers
        self.emb_dim = embd_dim
        self.vocab = vocab

        self.embd = nn.Embedding(self.vocab, self.emb_dim)

        self.rnn_layer = nn.RNN(
            input_size = self.emb_dim, 
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True)
        
        self.final_layer = nn.Linear(self.hidden_size, 2)

    def forward(self, input_text, targets):

            #Initializing h0
        h0 = torch.zeros(self.num_layers, input_text.size(0), self.hidden_size).to('cuda:0')
        out = self.embd(input_text)
        out,_ = self.rnn_layer(out, h0)
        out = self.final_layer(out[:,-1,:])

        loss = loss_fn(out,targets)

        pred = out.argmax(axis = 1)
                

        return pred, loss 