import torch
import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, in_size, hidden_size, out_size) -> None:
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(in_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(in_size + hidden_size, out_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        conc = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(conc)
        output = self.i2o(conc)
        output = self.softmax(output)

        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)








