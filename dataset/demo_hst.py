from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.utils.data

from dataset import make_hierarchical_signals
from model import RecurrentNetContinual

import matplotlib.pyplot as plt


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    model = RecurrentNetContinual(n_in=200, n_hid=args.network_size, n_out=1,
                                  t_constant=args.t_constant, use_cuda=use_cuda).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    signals = []
    targets = []
    for i in range(1):
        signal, target = make_hierarchical_signals.hierarchical_signals(n_episodes=args.each_episodes,
                                                                        stim_dur=args.stim_dur,
                                                                        sig1_stim_dur=args.stim_dur,
                                                                        resp_dur=args.resp_dur,
                                                                        each_episodes=args.each_episodes,
                                                                        spon_rate=0.01)
        signals.append(signal)
        targets.append(target)

    signals = np.array(signals)
    targets = np.array(targets)

    signals = torch.from_numpy(signals)
    targets = torch.from_numpy(targets)

    hidden = torch.zeros(1, args.network_size, requires_grad=False)
    hidden = hidden.to(device)

    signals = signals.float()
    targets = targets.float()

    signals, targets = signals.to(device), targets.to(device)

    _, output, hidden = model(signals, hidden)

    plt.figure()
    plt.plot(targets[0].data.numpy().T[0], label='target')
    plt.plot(output[0].data.numpy().T[0], label='output')
    plt.legend()
    plt.savefig('result_epoch_{}_short.png'.format(args.model_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--n_stim', type=int, default=3)
    parser.add_argument('--each_episodes', type=int, default=5)
    parser.add_argument('--stim_dur', type=int, default=15)
    parser.add_argument('--resp_dur', type=int, default=10)
    parser.add_argument('--t_constant', type=float, default=0.2)
    parser.add_argument('--network_size', type=int, default=500)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_epoch', type=str)
    args = parser.parse_args()
    print(args)
    main()
