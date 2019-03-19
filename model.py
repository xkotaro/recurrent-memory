import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class RecurrentNet(nn.Module):
    def __init__(self, n_in, n_out, n_hid):
        super(RecurrentNet, self).__init__()
        self.n_hid = n_hid
        self.in_layer = nn.Linear(n_in, n_hid)
        self.hid_layer = nn.RNNCell(n_hid, n_hid, nonlinearity='relu')
        self.out_layer = nn.Linear(n_hid, n_out)

    def forward(self, input_signal, length):
        hidden_list = []
        output_list = []
        num_batch = input_signal.size(0)
        hidden = Variable(torch.zeros(num_batch, self.n_hid).type_as(input_signal.data))
        for t in range(length):
            x = F.relu(self.in_layer(input_signal[t]))
            hidden = F.relu(self.hid_layer(x, hidden))
            output = self.out_layer(hidden)
            hidden_list.append(hidden)
            output_list.append(output)

        return hidden_list, output_list
