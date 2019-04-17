from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import torch.nn.functional as F
from model.model import RecurrentNetTimeVariable, RecurrentNetTimeFixed

when_slow_signal = []
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
            when_slow_signal.append(episode)
            # accum_signal = np.pi * np.random.rand(1)
            accum_signal = slow_signal_values[episode//each_episodes]
            # slow_signal_values.append(accum_signal)
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
    use_cuda = False
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    if not args.time_fixed:
        model = RecurrentNetTimeVariable(n_in=200, n_hid=500, n_out=1,
                                         use_cuda=use_cuda).to(device)
    else:
        model = RecurrentNetTimeFixed(n_in=200, n_hid=500, n_out=1,
                                      use_cuda=use_cuda).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    signals = []
    targets = []
    for i in range(1):
        signal, target = lsm_signals(n_episodes=21,
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

    hidden_list, output, hidden = model(signals, hidden)

    if not args.only_fast_dynamics:
        X = hidden_list.data.numpy()[0]
    else:
        const_one = torch.Tensor([1])
        if args.time_fixed:
            alpha = model.alpha(const_one)
        else:
            alpha = F.sigmoid(model.alpha(const_one))
        thresholded_index = [i for i in range(500) if alpha[i] > args.threshold_time_scale]
        only_fast_dynamics = np.array([hidden_list.data.numpy()[0].T[i] for i in thresholded_index])
        X = only_fast_dynamics.T
    pca = PCA(n_components=3)
    pca.fit(X)

    Xd = pca.transform(X)

    # 色の設定
    cmap = plt.get_cmap("tab10")
    fig = plt.figure()
    ax = Axes3D(fig)

    # 軸ラベルの設定
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    color = 0
    for i in range(len(when_slow_signal) - 1):
        ax.scatter(Xd.T[0][36 * when_slow_signal[i]:36 * when_slow_signal[i + 1]],
                   Xd.T[1][36 * when_slow_signal[i]:36 * when_slow_signal[i + 1]],
                   Xd.T[2][36 * when_slow_signal[i]:36 * when_slow_signal[i + 1]],
                   color=cmap(i), label='{:.3f}'.format(slow_signal_values[i]))
        color += 1

    ax.scatter(Xd.T[0][36 * when_slow_signal[-1]:], Xd.T[1][36 * when_slow_signal[-1]:],
               Xd.T[2][36 * when_slow_signal[-1]:], color=cmap(color), label='{:.3f}'.format(slow_signal_values[-1]))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize pca space of internal dynamics.')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--time_fixed', action='store_true')
    parser.add_argument('--only_fast_dynamics', action='store_true')
    parser.add_argument('--threshold_time_scale', type=float, default=0.15)
    args = parser.parse_args()
    main()
