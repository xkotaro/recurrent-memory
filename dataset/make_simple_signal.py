import numpy as np


def simple_signals(n_episodes=100, n_in=100, stim_dur=15,
                   total_dur=36, each_episodes=10, kappa=5.0, spon_rate=0.08, n_stim=3):
    phi = np.linspace(0, np.pi, n_in)
    n_loc = 1
    nneuron = n_in * n_loc

    G1 = (3.0 / stim_dur) * np.random.choice([1.0], 1)
    G1 = np.repeat(G1, n_in, axis=0).T
    G1 = np.tile(G1, (stim_dur, 1))

    # signal & target
    a = np.random.poisson(0.8, n_episodes)
    Rs1 = []
    accum_signal = np.pi * np.random.rand(1)
    target_list = []

    for episode in range(n_episodes):
        target_list.append(np.zeros(stim_dur * n_stim))
        if a[episode] == 2 or episode % each_episodes == 0:
            accum_signal = np.pi * np.random.rand(1)
            S = np.repeat(accum_signal, n_in, axis=0).T
            S = np.tile(S, (stim_dur, 1))

            L = G1 * np.exp(kappa * (np.cos(
                2.0 * (S - np.tile(phi, (stim_dur, n_loc)))) - 1.0))  # stim
            R = np.random.poisson(L)
            Rs1.append(R)
        else:
            Lr = (spon_rate / stim_dur) * np.ones((stim_dur, nneuron))  # resp
            R = np.random.poisson(Lr)
            Rs1.append(R)
        target = np.repeat(0, stim_dur, axis=0)
        target_list.append(target)
        L_spont = (spon_rate / (total_dur-stim_dur)) * np.ones((total_dur - stim_dur, nneuron))  # resp
        R = np.random.poisson(L_spont)
        Rs1.append(R)

        target = np.repeat(accum_signal, total_dur-stim_dur, axis=0)
        target_list.append(target)

    signal = np.concatenate(tuple(Rs1), axis=0)

    target = np.concatenate(tuple(target_list), axis=0)
    target = np.expand_dims(target, 1)

    return signal, target
