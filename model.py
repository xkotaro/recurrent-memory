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
        self.hid_layer = nn.RNNCell(n_hid, n_hid, nonlinearity='relu')
        self.out_layer = nn.Linear(n_hid, n_out)

    def forward(self, input_signal):
        num_batch = input_signal.size(0)
        length = input_signal.size(1)
        hidden = torch.zeros(num_batch, self.n_hid, requires_grad=True).type_as(input_signal.data)
        hidden_list = torch.zeros(length, num_batch, self.n_hid, requires_grad=True).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out, requires_grad=True).type_as(input_signal.data)
        input_signal = input_signal.permute(1, 0, 2)
        # print(input_signal.shape)
        alpha = torch.Tensor([0.02])
        alpha = alpha.to('cuda')
        for t in range(length):
            x = self.in_layer(input_signal[t])
            hidden = (1-alpha) * x + alpha * self.hid_layer(x, hidden)
            output = self.out_layer(hidden)
            hidden_list[t] = hidden
            output_list[t] = output
        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        # print(output_list.shape)
        return hidden_list, output_list
