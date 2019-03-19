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
        self.hid_layer = nn.RNN(n_hid, n_hid, nonlinearity='relu')
        self.out_layer = nn.Linear(n_hid, n_out)

    def forward(self, input_signal):
        # num_batch = input_signal.size(0)
        # length = input_signal.size(1)
        # hidden = torch.zeros(num_batch, self.n_hid, requires_grad=True).type_as(input_signal.data)
        # hidden_list = torch.zeros(length, num_batch, self.n_hid, requires_grad=True).type_as(input_signal.data)
        # output_list = torch.zeros(length, num_batch, self.n_out, requires_grad=True).type_as(input_signal.data)
        # input_signal = input_signal.permute(1, 0, 2)
        # print(input_signal.shape)
        x = self.in_layer(input_signal)
        # print("x.shape: ", x.shape)
        x = x.permute(1, 0, 2)
        hidden = self.hid_layer(x)
        # print("hidden.shape: ", hidden[0].shape)
        output = self.out_layer(hidden[0])

        # print(output.shape)
        output = output.permute(1, 0, 2)
        return hidden, output
