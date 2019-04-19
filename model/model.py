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
        alpha = torch.Tensor([self.t_constant])
        alpha = alpha.to('cuda')
        for t in range(length):
            x = self.in_layer(input_signal[t])
            hidden = (1 - alpha) * hidden + alpha * self.hid_layer(x, hidden)
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
        hidden_list = torch.zeros(length, num_batch, self.n_hid, requires_grad=True).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out, requires_grad=True).type_as(input_signal.data)
        input_signal = input_signal.permute(1, 0, 2)
        alpha = torch.Tensor([self.t_constant])
        if self.use_cuda:
            alpha = alpha.to('cuda')

        for t in range(length):
            x = self.in_layer(input_signal[t])
            hidden = (1 - alpha) * hidden + alpha * self.hid_layer(x, hidden)
            output = self.out_layer(hidden)
            hidden_list[t] = hidden
            output_list[t] = output
        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        return hidden_list, output_list, hidden


class RecurrentNetTimeVariableOld(nn.Module):
    def __init__(self, n_in, n_out, n_hid, use_cuda):
        super(RecurrentNetTimeVariableOld, self).__init__()
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
            hidden = (1 - alpha) * hidden + alpha * self.hid_layer(x, hidden)
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
        alpha_weight = np.array([[0.3]] * 500)

        if use_cuda:
            self.alpha.weight = torch.nn.Parameter(torch.from_numpy(alpha_weight).float().to('cuda'))
        else:
            self.alpha.weight = torch.nn.Parameter(torch.from_numpy(alpha_weight).float().to('cpu'))

        for param in self.in_layer.parameters():
            param.requires_grad = False

        for param in self.out_layer.parameters():
            param.requires_grad = False

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
            hidden = (1 - alpha) * hidden + alpha * self.hid_layer(x, hidden)
            output = self.out_layer(hidden)
            hidden_list[t] = hidden
            output_list[t] = output
        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        return hidden_list, output_list, hidden


class RecurrentNetTimeFixed(nn.Module):
    def __init__(self, n_in, n_out, n_hid, use_cuda, alpha_weight=np.array([0.1] * 50 + [0.4] * 450)):
        super(RecurrentNetTimeFixed, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.in_layer = nn.Linear(n_in, n_hid)
        self.hid_layer = nn.RNNCell(n_hid, n_hid, nonlinearity='relu')
        self.out_layer = nn.Linear(n_hid, n_out)
        self.use_cuda = use_cuda
        self.alpha = nn.Linear(1, n_hid, bias=False)
        """
        alpha_weight = np.array([0.5027, 0.4870, 0.2169, 0.6417, 0.6296, 0.3323, 0.3657, 0.3103, 0.7358,
        0.4409, 0.6844, 0.4695, 0.5713, 0.5025, 0.4282, 0.3991, 0.6625, 0.5481,
        0.3860, 0.9180, 0.5853, 0.4382, 0.6517, 0.2430, 0.4231, 0.3124, 0.2862,
        0.6124, 0.6178, 0.6217, 0.5928, 0.6674, 0.4408, 0.6559, 0.7671, 0.4456,
        0.6640, 0.5837, 0.7572, 0.3751, 0.2839, 0.5961, 0.3923, 0.1912, 0.6382,
        0.0724, 0.4589, 0.3514, 0.6515, 0.1211, 0.3374, 0.6320, 0.2724, 0.8581,
        0.7003, 0.2884, 0.2236, 0.5115, 0.4825, 0.3360, 0.6619, 0.3078, 0.6887,
        0.5888, 0.4273, 0.7264, 0.2996, 0.6810, 0.5525, 0.6951, 0.3222, 0.3372,
        0.2240, 0.6961, 0.2846, 0.5877, 0.2428, 0.7062, 0.1982, 0.2119, 0.2906,
        0.2676, 0.3626, 0.7122, 0.4723, 0.2079, 0.4546, 0.2495, 0.4690, 0.5310,
        0.5478, 0.2465, 0.7133, 0.6816, 0.1910, 0.1976, 0.3309, 0.3605, 0.1462,
        0.4253, 0.4720, 0.2742, 0.3733, 0.7162, 0.7297, 0.3716, 0.5880, 0.4421,
        0.2984, 0.6722, 0.5931, 0.5445, 0.1726, 0.6889, 0.8273, 0.6474, 0.2992,
        0.3905, 0.5377, 0.3106, 0.6854, 0.6883, 0.5056, 0.5173, 0.6131, 0.3893,
        0.5935, 0.8289, 0.6765, 0.3230, 0.7139, 0.3388, 0.3623, 0.3493, 0.1611,
        0.4815, 0.6313, 0.6961, 0.5940, 0.5769, 0.3361, 0.5600, 0.5143, 0.2875,
        0.5450, 0.4471, 0.5711, 0.1804, 0.3099, 0.8265, 0.5610, 0.4987, 0.4816,
        0.3492, 0.5617, 0.7465, 0.4613, 0.1895, 0.5141, 0.6781, 0.5539, 0.7658,
        0.3715, 0.5626, 0.2743, 0.4118, 0.4056, 0.5546, 0.3474, 0.4778, 0.4645,
        0.5637, 0.5080, 0.4704, 0.5987, 0.4833, 0.4594, 0.3115, 0.2895, 0.2942,
        0.7953, 0.6292, 0.6262, 0.5487, 0.7310, 0.2876, 0.2428, 0.3399, 0.5955,
        0.6914, 0.6837, 0.7394, 0.1599, 0.4653, 0.2267, 0.3179, 0.7249, 0.3359,
        0.3574, 0.7770, 0.6992, 0.2763, 0.4542, 0.6685, 0.5128, 0.5738, 0.2861,
        0.2968, 0.2867, 0.7166, 0.4496, 0.5339, 0.7651, 0.6067, 0.2360, 0.7668,
        0.3362, 0.5552, 0.5838, 0.3247, 0.6148, 0.4945, 0.3623, 0.7240, 0.1641,
        0.6290, 0.5067, 0.4285, 0.5233, 0.3591, 0.4078, 0.8611, 0.1626, 0.3174,
        0.1785, 0.3162, 0.4170, 0.2246, 0.3716, 0.4540, 0.6327, 0.5199, 0.4903,
        0.5660, 0.4528, 0.5297, 0.5249, 0.6117, 0.4529, 0.6611, 0.3920, 0.6055,
        0.4267, 0.4016, 0.2550, 0.3627, 0.6336, 0.7089, 0.4518, 0.8501, 0.3122,
        0.4434, 0.5503, 0.7002, 0.6853, 0.3608, 0.2394, 0.6333, 0.6091, 0.3028,
        0.4728, 0.3303, 0.6432, 0.2525, 0.6200, 0.2829, 0.7160, 0.5704, 0.5765,
        0.2138, 0.6169, 0.5776, 0.3603, 0.3489, 0.4775, 0.7895, 0.6323, 0.6096,
        0.5327, 0.4837, 0.3388, 0.4689, 0.5485, 0.5767, 0.2990, 0.5748, 0.4084,
        0.6130, 0.3141, 0.6591, 0.7066, 0.4655, 0.2475, 0.2921, 0.2446, 0.6039,
        0.5616, 0.6280, 0.2775, 0.5149, 0.5778, 0.2617, 0.0738, 0.1922, 0.6558,
        0.3669, 0.2187, 0.6953, 0.2620, 0.5450, 0.1670, 0.4645, 0.6338, 0.5361,
        0.4267, 0.0548, 0.2076, 0.7396, 0.4577, 0.3222, 0.6883, 0.3999, 0.3482,
        0.3579, 0.4462, 0.5579, 0.4339, 0.3390, 0.2138, 0.4453, 0.7362, 0.5550,
        0.6432, 0.3246, 0.5915, 0.2974, 0.1351, 0.3065, 0.1870, 0.6937, 0.6081,
        0.4459, 0.6639, 0.6686, 0.2951, 0.4051, 0.2302, 0.3926, 0.2727, 0.6519,
        0.2334, 0.7136, 0.7340, 0.3673, 0.2270, 0.3232, 0.3468, 0.2755, 0.6595,
        0.3370, 0.6108, 0.2796, 0.0891, 0.3039, 0.4202, 0.5286, 0.6412, 0.2919,
        0.3824, 0.3077, 0.7550, 0.5020, 0.2782, 0.2683, 0.5891, 0.4082, 0.5516,
        0.5483, 0.4080, 0.4601, 0.4895, 0.3233, 0.4305, 0.3626, 0.4071, 0.3324,
        0.1275, 0.4523, 0.1837, 0.2921, 0.3327, 0.4915, 0.2329, 0.6102, 0.2036,
        0.3200, 0.3721, 0.3622, 0.4516, 0.2492, 0.4104, 0.5895, 0.2737, 0.6579,
        0.4438, 0.7426, 0.6949, 0.6045, 0.4873, 0.6837, 0.7022, 0.2181, 0.2598,
        0.2261, 0.4763, 0.6051, 0.3479, 0.6023, 0.3788, 0.1466, 0.5097, 0.5534,
        0.5943, 0.4652, 0.4894, 0.6619, 0.1341, 0.7212, 0.3109, 0.6395, 0.5900,
        0.1652, 0.2392, 0.4761, 0.5349, 0.1548, 0.6869, 0.6224, 0.4814, 0.5237,
        0.5377, 0.2130, 0.6760, 0.2566, 0.0777, 0.3668, 0.5607, 0.4975, 0.5887,
        0.9304, 0.7369, 0.6626, 0.4307, 0.5429, 0.3632, 0.3022, 0.4446, 0.3853,
        0.5009, 0.5759, 0.2127, 0.2673, 0.3153, 0.5106, 0.5318, 0.3882, 0.2723,
        0.8020, 0.7562, 0.5263, 0.3956, 0.6900, 0.4912, 0.4562, 0.6552, 0.4804,
        0.6823, 0.5021, 0.3416, 0.5377, 0.6399, 0.1425, 0.5595, 0.2267, 0.6512,
        0.1502, 0.5517, 0.4139, 0.6237, 0.2281])
        """
        alpha_weight = np.expand_dims(alpha_weight, axis=1)
        if use_cuda:
            self.alpha.weight = torch.nn.Parameter(torch.from_numpy(alpha_weight).float().to('cuda'))
        else:
            self.alpha.weight = torch.nn.Parameter(torch.from_numpy(alpha_weight).float().to('cpu'))

        for param in self.in_layer.parameters():
            param.requires_grad = False

        for param in self.out_layer.parameters():
            param.requires_grad = False

        for param in self.alpha.parameters():
            param.requires_grad = False

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
            hidden = (1 - alpha) * hidden + alpha * self.hid_layer(x, hidden)
            output = self.out_layer(hidden)
            hidden_list[t] = hidden
            output_list[t] = output
        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        return hidden_list, output_list, hidden


class RecurrentNetTimeFixedOnlyRec(nn.Module):
    def __init__(self, n_in, n_out, n_hid, use_cuda, alpha_weight=np.array([0.1] * 50 + [0.4] * 450)):
        super(RecurrentNetTimeFixedOnlyRec, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        # self.in_layer = nn.Linear(n_in, n_hid)
        self.hid_layer = nn.RNNCell(n_in, n_hid, nonlinearity='relu')
        self.out_layer = nn.Linear(n_hid, n_out)
        self.use_cuda = use_cuda
        self.alpha = nn.Linear(1, n_hid, bias=False)

        alpha_weight = np.expand_dims(alpha_weight, axis=1)
        if use_cuda:
            self.alpha.weight = torch.nn.Parameter(torch.from_numpy(alpha_weight).float().to('cuda'))
        else:
            self.alpha.weight = torch.nn.Parameter(torch.from_numpy(alpha_weight).float().to('cpu'))

        # tmp = 0
        # for param in self.hid_layer.parameters():
        #     if tmp == 0 or tmp == 2:
        #         param.requires_grad = False
        #     tmp += 1

        for param in self.out_layer.parameters():
            param.requires_grad = False

        for param in self.alpha.parameters():
            param.requires_grad = False

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
            hidden = (1 - alpha) * hidden + alpha * self.hid_layer(input_signal[t], hidden)
            output = self.out_layer(hidden)
            hidden_list[t] = hidden
            output_list[t] = output
        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        return hidden_list, output_list, hidden
