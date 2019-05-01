from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.utils.data

from model.model import RecurrentNetTimeFixed


def lsm_signals(n_episodes=100, n_in=100, stim_dur=15,
                sig1_stim_dur=20, resp_dur=10, each_episodes=10, kappa=5.0, spon_rate=0.2, n_stim=3):
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
            # print(episode)
            accum_signal = np.pi * np.random.rand(1)
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


def main():
    batch_size = args.batch_size
    # total_length = args.total_length
    each_episodes = args.each_episodes
    spon_rate = args.spon_rate
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    # print(device)
    model = RecurrentNetTimeFixed(n_in=200, n_hid=500, n_out=1,
                                  use_cuda=use_cuda).to(device)

    n_stim=3
    stim_dur = 7

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    signals = []
    targets = []
    for i in range(batch_size):
        signal, target = lsm_signals(n_episodes=each_episodes * batch_size,
                                     stim_dur=7,
                                     sig1_stim_dur=7,
                                     resp_dur=5,
                                     each_episodes=each_episodes,
                                     spon_rate=spon_rate)

        signals.append(signal)
        targets.append(target)
    signals = np.array(signals)
    targets = np.array(targets)

    signals = torch.from_numpy(signals)
    targets = torch.from_numpy(targets)

    hidden = torch.zeros(batch_size, 500, requires_grad=False)
    hidden = hidden.to(device)
    total_loss = 0
    one_learning_length = 3 * (5 + 7)
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

        hidden = hidden.detach()
        hidden_list, output, hidden = model(batched_signals, hidden)

        for i in range(each_episodes):
            loss = torch.nn.MSELoss()(output[:, n_stim * stim_dur + (i + 1) * one_learning_length:
                                                 one_learning_length * (i + 2), :],
                                       batched_targets[:, n_stim * stim_dur + (i + 1) * one_learning_length:
                                                          one_learning_length * (i + 2), :])
            total_loss += loss.item()
            print(total_loss)
    print(total_loss/(each_episodes*batch_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-e', '--each_episodes', type=int)
    parser.add_argument('-s', '--spon_rate', type=float)
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    main()
