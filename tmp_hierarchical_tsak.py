from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data

import make_hierarchical_signals
from model import RecurrentNetContinual


def train(model, device, optimizer, resp_dur, n_stim, epoch, batch_size, n_hid):
    model.train()
    signals = []
    targets = []
    for i in range(batch_size):
        signal, target = make_hierarchical_signals.hierarchical_signals(n_episodes=200, spon_rate=0.01)
        signals.append(signal)
        targets.append(target)

    signals = np.array(signals)
    targets = np.array(targets)

    signals = torch.from_numpy(signals)
    targets = torch.from_numpy(targets)

    for two_episodes in range(100):
        batched_signals = signals[:, two_episodes*150:(two_episodes+1)*150, :]
        batched_targets = targets[:, two_episodes*150:(two_episodes+1)*150, :]

        batched_signals = batched_signals.float()
        batched_targets = batched_targets.float()
        batched_signals.requires_grad = True
        batched_signals, batched_targets = batched_signals.to(device), batched_targets.to(device)

        optimizer.zero_grad()
        hidden = torch.zeros(batch_size, n_hid, requires_grad=True)
        hidden = hidden.to(device)
        _, output, hidden = model(batched_signals, hidden)

        # loss = torch.nn.MSELoss()(output[:, 45:75, :], batched_targets[:, 45:75, :])
        loss = torch.nn.MSELoss()(output[:, 120:150, :], batched_targets[:, 120:150, :])
        loss.backward()
        optimizer.step()
        print("target: ", end=" ")
        for i in range(n_stim, 0, -1):
            print(batched_targets.cpu().data[0][-int(resp_dur * i)].numpy()[0], end=" ")
        print('\n')
        print("output: ", end=" ")
        for i in range(n_stim, 0, -1):
            print(output.cpu().data[0][-int(resp_dur * i)].numpy()[0], end=" ")
        print("\n")
        print('Train Epoch: {}, Episode: {}, Loss: {:.6f}'.format(
            epoch, two_episodes, loss.item()))


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    resp_dur = args.resp_dur

    model = RecurrentNetContinual(n_in=200, n_hid=args.network_size, n_out=1, t_constant=args.t_constant).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(model, device, optimizer, resp_dur, args.n_stim, epoch, args.batch_size, args.network_size)

    if args.save_model:
        torch.save(model.state_dict(), "recurrent_memory.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--n_stim', type=int, default=3)
    parser.add_argument('--resp_dur', type=int, default=10)
    parser.add_argument('--t_constant', type=float, default=0.2)
    parser.add_argument('--network_size', type=int, default=500)
    parser.add_argument('--test_batch_size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 50)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    print(args)
    main()
