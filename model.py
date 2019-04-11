import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RecurrentNet(nn.Module):
    def __init__(self, n_in, n_out, n_hid, t_constant):
        super(RecurrentNet, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.t_constant = t_constant
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
        alpha = torch.Tensor([self.t_constant])
        alpha = alpha.to('cuda')
        for t in range(length):
            x = self.in_layer(input_signal[t])
            hidden = (1-alpha) * hidden + alpha * self.hid_layer(x, hidden)
            output = self.out_layer(hidden)
            hidden_list[t] = hidden
            output_list[t] = output
        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        return hidden_list, output_list


class RecurrentNetContinual(nn.Module):
    def __init__(self, n_in, n_out, n_hid, t_constant, use_cuda):
        super(RecurrentNetContinual, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.t_constant = t_constant
        self.in_layer = nn.Linear(n_in, n_hid)
        self.hid_layer = nn.RNNCell(n_hid, n_hid, nonlinearity='relu')
        self.out_layer = nn.Linear(n_hid, n_out)
        self.use_cuda = use_cuda

    def forward(self, input_signal, hidden):
        num_batch = input_signal.size(0)
        length = input_signal.size(1)
        # hidden = torch.zeros(num_batch, self.n_hid, requires_grad=True).type_as(input_signal.data)
        hidden_list = torch.zeros(length, num_batch, self.n_hid, requires_grad=True).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out, requires_grad=True).type_as(input_signal.data)
        input_signal = input_signal.permute(1, 0, 2)
        # print(input_signal.shape)
        alpha = torch.Tensor([self.t_constant])
        if self.use_cuda:
            alpha = alpha.to('cuda')

        for t in range(length):
            x = self.in_layer(input_signal[t])
            hidden = (1-alpha) * hidden + alpha * self.hid_layer(x, hidden)
            output = self.out_layer(hidden)
            hidden_list[t] = hidden
            output_list[t] = output
        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        return hidden_list, output_list, hidden


class RecurrentNetTimeVariable(nn.Module):
    def __init__(self, n_in, n_out, n_hid, use_cuda):
        super(RecurrentNetTimeVariable, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.in_layer = nn.Linear(n_in, n_hid)
        self.hid_layer = nn.RNNCell(n_hid, n_hid, nonlinearity='relu')
        self.out_layer = nn.Linear(n_hid, n_out)
        self.use_cuda = use_cuda
        self.alpha = nn.Linear(1, n_hid, bias=False)

    def forward(self, input_signal, hidden):
        num_batch = input_signal.size(0)
        length = input_signal.size(1)
        hidden_list = torch.zeros(length, num_batch, self.n_hid, requires_grad=True).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out, requires_grad=True).type_as(input_signal.data)
        input_signal = input_signal.permute(1, 0, 2)

        for param in self.in_layer.parameters():
            param.requires_grad = False

        for param in self.out_layer.parameters():
            param.requires_grad = False
        const_one = torch.Tensor([1])
        if self.use_cuda:
            const_one = const_one.to('cuda')
        alpha = F.sigmoid(self.alpha(const_one))

        for t in range(length):
            x = self.in_layer(input_signal[t])
            hidden = (1-alpha) * hidden + alpha * self.hid_layer(x, hidden)
            output = self.out_layer(hidden)
            hidden_list[t] = hidden
            output_list[t] = output
        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        return hidden_list, output_list, hidden


class RecurrentNetTimeFixed(nn.Module):
    def __init__(self, n_in, n_out, n_hid, use_cuda):
        super(RecurrentNetTimeFixed, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.in_layer = nn.Linear(n_in, n_hid)
        self.hid_layer = nn.RNNCell(n_hid, n_hid, nonlinearity='relu')
        self.out_layer = nn.Linear(n_hid, n_out)
        self.use_cuda = use_cuda
        self.alpha = nn.Linear(1, n_hid, bias=False)
        alpha_weight = np.array([[0.2]] * 10 + [[0.5]] * 490)
        self.alpha.weight = torch.nn.Parameter(torch.from_numpy(alpha_weight).float().to('cuda'))

        for param in self.in_layer.parameters():
            param.requires_grad = False

        for param in self.out_layer.parameters():
            param.requires_grad = False

        # for param in self.alpha.parameters():
        #     param.requires_grad = False

    def forward(self, input_signal, hidden):
        num_batch = input_signal.size(0)
        length = input_signal.size(1)
        hidden_list = torch.zeros(length, num_batch, self.n_hid, requires_grad=True).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out, requires_grad=True).type_as(input_signal.data)
        input_signal = input_signal.permute(1, 0, 2)
        const_one = torch.Tensor([1])
        if self.use_cuda:
            const_one = const_one.to('cuda')
        alpha = self.alpha(const_one)

        for t in range(length):
            x = self.in_layer(input_signal[t])
            hidden = (1-alpha) * hidden + alpha * self.hid_layer(x, hidden)
            output = self.out_layer(hidden)
            hidden_list[t] = hidden
            output_list[t] = output
        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        return hidden_list, output_list, hidden
