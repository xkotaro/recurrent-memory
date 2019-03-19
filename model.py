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
        self.n_out = n_out
        self.in_layer = nn.Linear(n_in, n_hid)
        self.hid_layer = nn.RNNCell(n_hid, n_hid, nonlinearity='tanh')
        self.out_layer = nn.Linear(n_hid, n_out)

    def forward(self, input_signal):
        num_batch = input_signal.size(0)
        length = input_signal.size(1)
        hidden = Variable(torch.zeros(num_batch, self.n_hid).type_as(input_signal.data))
        # print(hidden.requires_grad)
        hidden_list = Variable(torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data))
        output_list = Variable(torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data))
        input_signal = input_signal.permute(1, 0, 2)
        # print(input_signal.shape)
        for t in range(length):
            x = F.relu(self.in_layer(input_signal[t]))
            hidden = F.relu(self.hid_layer(x, hidden))
            output = self.out_layer(hidden)
            hidden_list[t] = hidden
            output_list[t] = output
        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        # print(output_list.shape)
        return hidden_list, output_list
