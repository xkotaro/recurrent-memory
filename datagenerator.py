import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.misc import comb
import scipy.stats as scistat


class DelayedEstimationTask(Dataset):
    """Parameters"""

    def __init__(self, max_iter=None, n_loc=1, n_in=25, n_out=1, stim_dur=10, delay_dur=100, resp_dur=10,
                 kappa=2.0, spon_rate=0.001, transform=None):
        super(DelayedEstimationTask, self).__init__()
        self.n_in = n_in  # number of neurons per location
        self.n_out = n_out
        self.n_loc = n_loc
        self.kappa = kappa
        self.spon_rate = spon_rate
        self.nneuron = self.n_in * self.n_loc  # total number of input neurons
        self.phi = np.linspace(0, np.pi, self.n_in)
        self.stim_dur = stim_dur
        self.delay_dur = delay_dur
        self.resp_dur = resp_dur
        self.total_dur = stim_dur + delay_dur + resp_dur
        self.max_iter = max_iter
        self.transform = transform

    def __len__(self):
        return self.max_iter

    def __getitem__(self, item):
        G = (1.0 / self.stim_dur) * np.random.choice([1.0], self.n_loc)
        G = np.repeat(G, self.n_in, axis=0).T
        G = np.tile(G, (self.stim_dur, 1))
        G = np.swapaxes(G, 0, 1)

        S1 = np.pi * np.random.rand(self.n_loc)

        S = S1.copy()  # S: signal
        S1 = np.repeat(S1, self.n_in, axis=0).T
        S1 = np.tile(S1, (self.stim_dur, 1, 1))
        S1 = np.swapaxes(S1, 0, 1)

        # Noisy responses
        L1 = G * np.exp(self.kappa * (np.cos(
            2.0 * (S1 - np.tile(self.phi, (self.stim_dur, self.n_loc)))) - 1.0))  # stim
        Ld = (self.spon_rate / self.delay_dur) * np.ones((self.delay_dur, self.nneuron))  # delay
        Lr = (self.spon_rate / self.resp_dur) * np.ones((self.resp_dur, self.nneuron))  # resp

        R1 = np.random.poisson(L1)
        Rd = np.random.poisson(Ld)
        Rr = np.random.poisson(Lr)

        example_input = np.concatenate((R1, Rd, Rr), axis=1)
        example_output = np.repeat(S[:, np.newaxis, :], self.total_dur, axis=1)

        cum_R1 = np.sum(R1, axis=1)
        mu_x = np.asarray([np.arctan2(np.dot(cum_R1[:, i * self.n_in:(i + 1) * self.n_in], np.sin(2.0 * self.phi)),
                                      np.dot(cum_R1[:, i * self.n_in:(i + 1) * self.n_in], np.cos(2.0 * self.phi))) for
                           i in range(self.n_loc)]) / 2.0
        mu_x = (mu_x > 0.0) * mu_x + (mu_x < 0.0) * (mu_x + np.pi)
        mu_x = mu_x.T

        if self.transform:
            example_input = self.transform(example_input)

        return example_input, example_output, S, mu_x