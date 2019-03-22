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
    def __init__(self, max_iter=None, n_loc=1, n_in=25, stim_dur=10, delay_dur=100, resp_dur=10,
                 kappa=2.0, spon_rate=0.1, each_stim_dur=15, transform=None):
        super(RepeatSignals, self).__init__()
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
        G = (1.0 / self.each_stim_dur) * np.random.choice([1.0], self.n_loc)
        G = np.repeat(G, self.n_in, axis=0).T
        G = np.tile(G, (self.each_stim_dur, 1))

        S1 = np.pi * np.random.rand(self.n_loc)
        S2 = np.pi * np.random.rand(self.n_loc)
        S3 = np.pi * np.random.rand(self.n_loc)

        S_1 = S1.copy()  # S: signal
        S1 = np.repeat(S1, self.n_in, axis=0).T
        S1 = np.tile(S1, (self.each_stim_dur, 1))

        S_2 = S2.copy()  # S: signal
        S2 = np.repeat(S2, self.n_in, axis=0).T
        S2 = np.tile(S2, (self.each_stim_dur, 1))

        S_3 = S3.copy()  # S: signal
        S3 = np.repeat(S3, self.n_in, axis=0).T
        S3 = np.tile(S3, (self.each_stim_dur, 1))

        # Noisy responses
        L1 = G * np.exp(self.kappa * (np.cos(
            2.0 * (S1 - np.tile(self.phi, (self.each_stim_dur, self.n_loc)))) - 1.0))  # stim1
        L2 = G * np.exp(self.kappa * (np.cos(
            2.0 * (S2 - np.tile(self.phi, (self.each_stim_dur, self.n_loc)))) - 1.0))  # stim2
        L3 = G * np.exp(self.kappa * (np.cos(
            2.0 * (S3 - np.tile(self.phi, (self.each_stim_dur, self.n_loc)))) - 1.0))  # stim3
        Lr = (self.spon_rate / self.resp_dur) * np.ones((self.each_stim_dur, self.nneuron))  # resp

        R1 = np.random.poisson(L1)
        R2 = np.random.poisson(L2)
        R3 = np.random.poisson(L3)
        Rr = np.random.poisson(Lr)

        example_input = np.concatenate((R1, R2, R3, Rr), axis=0)
        # print(example_input.shape)
        target1 = np.repeat(S_1, self.each_stim_dur/3, axis=0)
        target2 = np.repeat(S_2, self.each_stim_dur/3, axis=0)
        target3 = np.repeat(S_3, self.each_stim_dur/3, axis=0)
        target = np.concatenate((np.zeros(self.each_stim_dur*3), target1, target2, target3), axis=0)
        target = np.expand_dims(target, 1)
        # print(example_output.shape)

        if self.transform:
            example_input = self.transform(example_input)
            target = self.transform(target)

        return example_input, target
