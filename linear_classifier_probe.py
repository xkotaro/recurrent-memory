import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from model.model import RecurrentNetTimeFixed

import torch.optim as optim

slow_signal_values = [1.278, 2.176, 0.451]


def lsm_signals(n_episodes=100, n_in=100, stim_dur=15,
                sig1_stim_dur=20, resp_dur=10, each_episodes=10, kappa=5.0, spon_rate=0.08, n_stim=3):
    phi = np.linspace(0, np.pi, n_in)
    n_loc = 1
    nneuron = n_in * n_loc
    total_dur = n_stim * (stim_dur + resp_dur)
    G = (1.0 / stim_dur) * np.random.choice([1.0], 1)
    G = np.repeat(G, n_in, axis=0).T
    G = np.tile(G, (stim_dur, 1))

    # signal2
    Stims = []
    Stims_ = []
    Ls = []
    Rs = []
    for episode in range(n_episodes):
        episode_stim = []
        for i in range(n_stim):
            S = np.pi * np.random.rand(1)
            S_ = S.copy()
            S = np.repeat(S, n_in, axis=0).T
            S = np.tile(S, (stim_dur, 1))
            Stims.append(S)
            episode_stim.append(S_)
            # Noisy responses
            L = G * np.exp(kappa * (np.cos(
                2.0 * (S - np.tile(phi, (stim_dur, n_loc)))) - 1.0))  # stim
            Ls.append(L)
            R = np.random.poisson(L)
            Rs.append(R)
        Stims_.append(episode_stim)
        Lr = (spon_rate / resp_dur) * np.ones((resp_dur * n_stim, nneuron))  # resp
        Rr = np.random.poisson(Lr)

        Rs.append(Rr)

    signal2 = np.concatenate(tuple(Rs), axis=0)

    G1 = (3.0 / sig1_stim_dur) * np.random.choice([1.0], 1)
    G1 = np.repeat(G1, n_in, axis=0).T
    G1 = np.tile(G1, (sig1_stim_dur, 1))

    # signal1 & target
    Rs1 = []
    accum_signal = np.pi * np.random.rand(1)
    target_list = []

    for episode in range(n_episodes):
        target_list.append(np.zeros(stim_dur * n_stim))
        if episode % each_episodes == 0:
            accum_signal = slow_signal_values[(episode // each_episodes) % 3]
            S = np.repeat(accum_signal, n_in, axis=0).T
            S = np.tile(S, (sig1_stim_dur, 1))

            L = G1 * np.exp(kappa * (np.cos(
                2.0 * (S - np.tile(phi, (sig1_stim_dur, n_loc)))) - 1.0))  # stim
            R = np.random.poisson(L)
            Rs1.append(R)
        else:
            Lr = (spon_rate / resp_dur) * np.ones((sig1_stim_dur, nneuron))  # resp
            R = np.random.poisson(Lr)
            Rs1.append(R)
        L_spont = (spon_rate / resp_dur) * np.ones((total_dur - sig1_stim_dur, nneuron))  # resp
        R = np.random.poisson(L_spont)
        Rs1.append(R)

        for i in range(n_stim):
            target = np.repeat(Stims_[episode][i] + accum_signal, resp_dur, axis=0)
            target_list.append(target)

    signal1 = np.concatenate(tuple(Rs1), axis=0)

    target = np.concatenate(tuple(target_list), axis=0)
    target = np.expand_dims(target, 1)

    signal = np.concatenate((signal1, signal2), axis=1)
    return signal, target


def make_dataset(n_episodes):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    # print(device)

    model = RecurrentNetTimeFixed(n_in=200, n_hid=500, n_out=1,
                                  use_cuda=use_cuda).to(device)
    """
    if device == 'cuda':
        model.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    """

    signals = []
    targets = []
    for i in range(1):
        signal, target = lsm_signals(n_episodes=n_episodes,
                                     stim_dur=7,
                                     sig1_stim_dur=7,
                                     resp_dur=5,
                                     each_episodes=7,
                                     spon_rate=0.01)
        signals.append(signal)
        targets.append(target)

    signals = np.array(signals)
    targets = np.array(targets)

    signals = torch.from_numpy(signals)
    targets = torch.from_numpy(targets)

    hidden = torch.zeros(1, 500, requires_grad=False)
    hidden = hidden.to(device)

    signals = signals.float()
    targets = targets.float()

    signals, targets = signals.to(device), targets.to(device)

    hidden_list, _, _ = model(signals, hidden)

    target_list = np.array([int((i // 12) % 3) for i in range(n_episodes*36)])
    # target_list = torch.from_numpy(target_list).long()

    return hidden_list[0].detach().numpy(), target_list


class OneLayerNetwork(nn.Module):
    def __init__(self, n_in, n_out):
        super(OneLayerNetwork, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, hidden_state):
        y = self.fc(hidden_state)
        return y


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.only_fast_dynamics:
        net = OneLayerNetwork(args.num_fast, 3).to(device)
    elif args.only_slow_dynamics:
        net = OneLayerNetwork(args.num_slow, 3).to(device)
    else:
        net = OneLayerNetwork(500, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(args.epochs):
        net.train()
        data, label = make_dataset(5)
        # print(data.shape)
        if args.only_fast_dynamics:
            data = data[:, -args.num_fast:]
        elif args.only_slow_dynamics:
            data = data[:, :args.num_slow]
        p = np.random.permutation(data.shape[0])
        data = data[p]
        label = label[p]
        # data.requires_grad = True
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        data = data.float()
        label = label.long()
        data.requires_grad = True
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        y = net(data)
        loss = criterion(y, label)
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            print('epoch: ', epoch, 'train loss: ', loss.item())
            correct = 0
            total = 5*36
            data, label = make_dataset(5)
            if args.only_fast_dynamics:
                data = data[:, -args.num_fast:]
            elif args.only_slow_dynamics:
                data = data[:, :args.num_slow]
            data = torch.from_numpy(data)
            label = torch.from_numpy(label)
            data = data.float()
            data = data.to(device)
            label = label.to(device)
            net.eval()
            outpus = net(data)
            _, predicted = torch.max(outpus.data, 1)
            correct += (predicted == label).sum().item()
            print('Accuracy: {:.2f} %'.format(100 * float(correct / total)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--only_fast_dynamics', action='store_true')
    parser.add_argument('--num_fast', type=int, default=455)
    parser.add_argument('--only_slow_dynamics', action='store_true')
    parser.add_argument('--num_slow', type=int, default=45)
    args = parser.parse_args()
    print(args)
    main()
    """
    data, label = make_dataset(1)
    print(data.shape)
    print(label.shape)
    p = np.random.permutation(data.shape[0])
    data = data[p]
    label = label[p]
    print(label)
    """

