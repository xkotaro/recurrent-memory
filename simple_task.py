"""記憶長とノイズに対する頑健性の間のトレードオフはどのような条件で生まれるのか？"""

import argparse
import os
from datetime import datetime

import numpy as np
import pytz
import torch
import torch.optim as optim
import torch.utils.data

from dataset import make_simple_signal
from model.model import RecurrentNet


def train(model, device, optimizer, stim_dur, each_episodes, total_dur, epoch, batch_size, n_hid):
    model.train()
    signals = []
    targets = []
    for i in range(batch_size):
        signal, target = make_simple_signal.simple_signals(n_episodes=500, n_in=100, stim_dur=stim_dur,
                                                           total_dur=total_dur, each_episodes=each_episodes,
                                                           spon_rate=0.01)
        signals.append(signal)
        targets.append(target)

    signals = np.array(signals)
    targets = np.array(targets)

    signals = torch.from_numpy(signals)
    targets = torch.from_numpy(targets)

    hidden = torch.zeros(batch_size, n_hid, requires_grad=True)
    hidden = hidden.to(device)
    for episodes in range(batch_size):
        batched_signals = \
            signals[:, episodes * total_dur * each_episodes:
                       (episodes + 1) * total_dur * each_episodes, :]
        batched_targets = \
            targets[:, episodes * total_dur * each_episodes:
                       (episodes + 1) * total_dur * each_episodes, :]

        batched_signals = batched_signals.float()
        batched_targets = batched_targets.float()
        batched_signals.requires_grad = True
        batched_signals, batched_targets = batched_signals.to(device), batched_targets.to(device)

        optimizer.zero_grad()
        hidden = hidden.detach()
        _, output, hidden = model(batched_signals, hidden)

        loss = torch.nn.MSELoss()(output[:, stim_dur:total_dur, :],
                                  batched_targets[:, stim_dur:total_dur, :])
        for i in range(each_episodes - 1):
            loss += torch.nn.MSELoss()(output[:, stim_dur + (i + 1) * total_dur:
                                                 total_dur * (i + 2), :],
                                       batched_targets[:, stim_dur + (i + 1) * total_dur:
                                                          total_dur * (i + 2), :])
        loss.backward()
        optimizer.step()
        print("target: ", end=" ")
        print(batched_targets.cpu().data[0][-1].numpy()[0], end=" ")
        print('\n')
        print("output: ", end=" ")
        print(output.cpu().data[0][-1].numpy()[0], end=" ")
        print('Train Epoch: {}, Episode: {}, Loss: {:.6f}'.format(
            epoch, episodes, loss.item()))


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    os.makedirs("./trained_models", exist_ok=True)

    # alpha = [0.08]*80+[0.4]*420
    alpha = [0.4] * args.network_size
    model = RecurrentNet(n_in=100, n_hid=args.network_size, n_out=1,
                         use_cuda=use_cuda, alpha_weight=alpha).to(device)
    if args.trained_model:
        model.load_state_dict(
            torch.load(args.trained_model))

    print(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(model=model, device=device, optimizer=optimizer, stim_dur=args.stim_dur, each_episodes=args.each_episodes,
              total_dur=args.total_dur, epoch=epoch, batch_size=args.batch_size,
              n_hid=args.network_size)

        if args.save_model and (epoch - 1) % args.savepoint == 0:
            time_stamp = datetime.strftime(datetime.now(pytz.timezone('Japan')), '%m%d%H%M')
            torch.save(
                model.state_dict(),
                "./trained_models/{}_simple_task_all_04_epoch_{}.pth"
                .format(time_stamp, epoch, args.model_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--each_episodes', type=int, default=5)
    parser.add_argument('--stim_dur', type=int, default=15)
    parser.add_argument('--total_dur', type=int, default=10)
    parser.add_argument('--network_size', type=int, default=500)
    parser.add_argument('--savepoint', type=int, default=10)
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
    parser.add_argument('--trained_model', type=str, default=None)
    parser.add_argument('--model_id', type=str, default='1')

    args = parser.parse_args()
    print(args)
    main()
