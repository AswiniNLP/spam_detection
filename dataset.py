import torch
import torch.nn as nn

class My_dataloader:

    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,item):

        input_text  = self.input[item]
        targets = self.target[item]

        return {
            "input_text": input_text.type(torch.LongTensor),
            "targets": targets.type(torch.LongTensor),
        }
