import numpy as np


def input_driven_long_signals(n_episodes=100, n_in=100, stim_dur=15,
                              resp_dur=10, kappa=5.0, spon_rate=0.08, n_stim=3):
    phi = np.linspace(0, np.pi, n_in)
    n_loc = 1
    nneuron = n_in * n_loc
    G = (1.0 / stim_dur) * np.random.choice([1.0], 1)
    G = np.repeat(G, n_in, axis=0).T
    G = np.tile(G, (stim_dur, 1))

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
            L = G * np.exp(kappa * (np.cos(
                2.0 * (S - np.tile(phi, (stim_dur, n_loc)))) - 1.0))
            Ls.append(L)
            R = np.random.poisson(L)
            Rs.append(R)
        Stims_.append(episode_stim)
        Lr = (spon_rate / resp_dur) * np.ones((resp_dur * n_stim, nneuron))  # resp
        Rr = np.random.poisson(Lr)

        Rs.append(Rr)

    signal = np.concatenate(tuple(Rs), axis=0)
    target_list = []

    for episode in range(n_episodes):
        how_long_ago = 3
        if episode >= how_long_ago and 2.0 <= Stims_[episode-how_long_ago][0] + Stims_[episode-how_long_ago][1] <= 3.0:
            for i in range(2):
                target = np.repeat(Stims_[episode-how_long_ago][i], resp_dur, axis=0)
                target_list.append(target)
            target_list.append(np.zeros(n_stim*stim_dur-2*resp_dur))
        else:
            target_list.append(np.zeros(stim_dur * n_stim))

        for i in range(n_stim):
            target = np.repeat(Stims_[episode][i], resp_dur, axis=0)
            target_list.append(target)

    target = np.concatenate(tuple(target_list), axis=0)
    target = np.expand_dims(target, 1)

    return signal, target
