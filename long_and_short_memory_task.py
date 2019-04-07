from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
import torch.utils.data

import pytz
import make_lsm_signals
from model import RecurrentNetContinual, RecurrentNetTimeVariable


def train(model, device, optimizer, stim_dur, each_episodes, resp_dur, n_stim, epoch, batch_size, n_hid):
    model.train()
    signals = []
    targets = []
    for i in range(batch_size):
        signal, target = make_lsm_signals.lsm_signals(n_episodes=500, stim_dur=stim_dur,
                                                      sig1_stim_dur=stim_dur, resp_dur=resp_dur,
                                                      each_episodes=each_episodes, spon_rate=0.01)
        signals.append(signal)
        targets.append(target)

    signals = np.array(signals)
    targets = np.array(targets)

    signals = torch.from_numpy(signals)
    targets = torch.from_numpy(targets)

    hidden = torch.zeros(batch_size, n_hid, requires_grad=True)
    hidden = hidden.to(device)
    one_learning_length = n_stim * (resp_dur + stim_dur)
    for episodes in range(batch_size):
        batched_signals = \
            signals[:, episodes * one_learning_length * each_episodes:
                       (episodes + 1) * one_learning_length * each_episodes, :]
        batched_targets = \
            targets[:, episodes * one_learning_length * each_episodes:
                       (episodes + 1) * one_learning_length * each_episodes, :]

        batched_signals = batched_signals.float()
        batched_targets = batched_targets.float()
        batched_signals.requires_grad = True
        batched_signals, batched_targets = batched_signals.to(device), batched_targets.to(device)

        optimizer.zero_grad()
        hidden = hidden.detach()
        _, output, hidden = model(batched_signals, hidden)

        loss = torch.nn.MSELoss()(output[:, n_stim * stim_dur:one_learning_length, :],
                                  batched_targets[:, n_stim * stim_dur:one_learning_length, :])
        for i in range(each_episodes - 1):
            loss += torch.nn.MSELoss()(output[:, n_stim * stim_dur + (i + 1) * one_learning_length:
                                                 one_learning_length * (i + 2), :],
                                       batched_targets[:, n_stim * stim_dur + (i + 1) * one_learning_length:
                                                          one_learning_length * (i + 2), :])
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
            epoch, episodes, loss.item()))


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    os.makedirs("./work", exist_ok=True)

    # model = RecurrentNetContinual(n_in=200, n_hid=args.network_size, n_out=1,
    #                               t_constant=args.t_constant, use_cuda=use_cuda).to(device)
    model = RecurrentNetTimeVariable(n_in=200, n_hid=args.network_size, n_out=1,
                                     use_cuda=use_cuda).to(device)
    # model.in_layer.requires_grad = False
    # model.out_layer.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(model=model, device=device, optimizer=optimizer, stim_dur=args.stim_dur, each_episodes=args.each_episodes,
              resp_dur=args.resp_dur, n_stim=args.n_stim, epoch=epoch, batch_size=args.batch_size,
              n_hid=args.network_size)

        if args.save_model and (epoch - 1) % 10 == 0:
            time_stamp = datetime.strftime(datetime.now(pytz.timezone('Japan')), '%m%d%H%M')
            torch.save(model.state_dict(),
                       "/root/trained_models/{}_lsmsignals_netsize_{}_stimdur_{}_nstim_{}_respdur_{}_epoch_{}.pth"
                       .format(time_stamp, args.network_size, args.stim_dur, args.n_stim, args.resp_dur, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--n_stim', type=int, default=3)
    parser.add_argument('--each_episodes', type=int, default=5)
    parser.add_argument('--stim_dur', type=int, default=15)
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
