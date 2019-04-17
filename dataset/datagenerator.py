import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.misc import comb
import scipy.stats as scistat


class DelayedEstimationTask(Dataset):
    """Parameters"""

    def __init__(self, max_iter=None, n_loc=1, n_in=25, stim_dur=10, delay_dur=100, resp_dur=10,
                 kappa=2.0, spon_rate=0.1, each_stim_dur=15, transform=None):
        super(DelayedEstimationTask, self).__init__()
        self.n_in = n_in  # number of neurons per location
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
        self.each_stim_dur = each_stim_dur

    def __len__(self):
        return self.max_iter

    def __getitem__(self, item):
        G = (1.0 / self.stim_dur) * np.random.choice([1.0], self.n_loc)
        G = np.repeat(G, self.n_in, axis=0).T
        G = np.tile(G, (self.stim_dur, 1))

        S1 = np.pi * np.random.rand(self.n_loc)

        S = S1.copy()  # S: signal
        S1 = np.repeat(S1, self.n_in, axis=0).T
        S1 = np.tile(S1, (self.stim_dur, 1))
        # print(S1.shape)

        # Noisy responses
        L1 = G * np.exp(self.kappa * (np.cos(
            2.0 * (S1 - np.tile(self.phi, (self.stim_dur, self.n_loc)))) - 1.0))  # stim
        Ld = (self.spon_rate / self.delay_dur) * np.ones((self.delay_dur, self.nneuron))  # delay
        Lr = (self.spon_rate / self.resp_dur) * np.ones((self.resp_dur, self.nneuron))  # resp

        R1 = np.random.poisson(L1)
        Rd = np.random.poisson(Ld)
        Rr = np.random.poisson(Lr)

        # print(R1.shape)

        example_input = np.concatenate((R1, Rd, Rr), axis=0)
        # print(example_input.shape)
        example_output = np.repeat(S, self.total_dur, axis=0)
        example_output = np.expand_dims(example_output, 1)
        # print(example_output.shape)

        if self.transform:
            example_input = self.transform(example_input)
            example_output = self.transform(example_output)

        return example_input, example_output, S


class RepeatSignals(Dataset):
    def __init__(self, max_iter=None, n_loc=1, n_in=25, resp_dur=10,
                 kappa=2.0, spon_rate=0.1, stim_dur=15, n_stim=3, transform=None):
        super(RepeatSignals, self).__init__()
        self.n_in = n_in  # number of neurons per location
        self.n_loc = n_loc
        self.kappa = kappa
        self.spon_rate = spon_rate
        self.nneuron = self.n_in * self.n_loc  # total number of input neurons
        self.phi = np.linspace(0, np.pi, self.n_in)
        self.stim_dur = stim_dur
        self.resp_dur = resp_dur
        self.max_iter = max_iter
        self.n_stim = n_stim
        self.transform = transform

    def __len__(self):
        return self.max_iter

    def __getitem__(self, item):
        G = (1.0 / self.stim_dur) * np.random.choice([1.0], self.n_loc)
        G = np.repeat(G, self.n_in, axis=0).T
        G = np.tile(G, (self.stim_dur, 1))

        Stims = []
        Stims_ = []
        Ls = []
        Rs = []
        for i in range(self.n_stim):
            S = np.pi * np.random.rand(self.n_loc)
            S_ = S.copy()
            S = np.repeat(S, self.n_in, axis=0).T
            S = np.tile(S, (self.stim_dur, 1))
            Stims.append(S)
            Stims_.append(S_)

            # Noisy responses
            L = G * np.exp(self.kappa * (np.cos(
                2.0 * (S - np.tile(self.phi, (self.stim_dur, self.n_loc)))) - 1.0))  # stim

            Ls.append(L)
            R = np.random.poisson(L)
            Rs.append(R)
        Lr = (self.spon_rate / self.resp_dur) * np.ones((self.resp_dur*self.n_stim, self.nneuron))  # resp
        Rr = np.random.poisson(Lr)

        Rs.append(Rr)

        example_input = np.concatenate(tuple(Rs), axis=0)
        # print(example_input.shape)
        target_list = [(np.zeros(self.stim_dur * self.n_stim))]
        for i in range(self.n_stim):
            target = np.repeat(Stims_[i], self.resp_dur, axis=0)
            target_list.append(target)
        target = np.concatenate(tuple(target_list), axis=0)
        target = np.expand_dims(target, 1)

        if self.transform:
            example_input = self.transform(example_input)
            target = self.transform(target)

        return example_input, target
