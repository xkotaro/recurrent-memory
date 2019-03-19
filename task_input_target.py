# -*- coding: utf-8 -*-
import numpy as np
from scipy.misc import comb
import scipy.stats as scistat
import lasagne.init
import seaborn as sns
import matplotlib.pyplot as plt


def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the given axis
    """
    b = np.random.random(a.shape)
    idx = np.argsort(b, axis=axis)
    shuffled = a[np.arange(a.shape[0])[:, None], idx]
    return shuffled


class Task(object):

    def __init__(self, max_iter=None, batch_size=1):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1), self.sample()
        else:
            raise StopIteration()

    def sample(self):
        raise NotImplementedError()


class DelayedEstimationTask(Task):
    """Parameters"""

    def __init__(self, max_iter=None, batch_size=50, n_loc=1, n_in=25, n_out=1, stim_dur=10, delay_dur=100, resp_dur=10,
                 kappa=2.0, spon_rate=0.001, tr_cond='all_gains'):
        super(DelayedEstimationTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in = n_in  # number of neurons per location
        self.n_out = n_out
        self.n_loc = n_loc
        self.kappa = kappa
        self.spon_rate = spon_rate
        self.nneuron = self.n_in * self.n_loc  # total number of input neurons
        self.phi = np.linspace(0, np.pi, self.n_in)
        # print('phi: ', self.phi)
        self.stim_dur = stim_dur
        self.delay_dur = delay_dur
        self.resp_dur = resp_dur
        self.total_dur = stim_dur + delay_dur + resp_dur
        self.tr_cond = tr_cond

    def sample(self):

        if self.tr_cond == 'all_gains':
            G = (1.0 / self.stim_dur) * np.random.choice([1.0], size=(self.n_loc, self.batch_size))
            G = np.repeat(G, self.n_in, axis=0).T
            G = np.tile(G, (self.stim_dur, 1, 1))
            G = np.swapaxes(G, 0, 1)
        else:
            G = (0.5 / self.stim_dur) * np.random.choice([1.0], size=(1, self.batch_size))
            G = np.repeat(G, self.n_in * self.n_loc, axis=0).T
            G = np.tile(G, (self.stim_dur, 1, 1))
            G = np.swapaxes(G, 0, 1)

        S1 = np.pi * np.random.rand(self.n_loc, self.batch_size)
        # S: signal
        S = S1.T
        S1 = np.repeat(S1, self.n_in, axis=0).T
        S1 = np.tile(S1, (self.stim_dur, 1, 1))
        S1 = np.swapaxes(S1, 0, 1)

        # Noisy responses
        L1 = G * np.exp(self.kappa * (np.cos(
            2.0 * (S1 - np.tile(self.phi, (self.batch_size, self.stim_dur, self.n_loc)))) - 1.0))  # stim
        Ld = (self.spon_rate / self.delay_dur) * np.ones((self.batch_size, self.delay_dur, self.nneuron))  # delay
        Lr = (self.spon_rate / self.resp_dur) * np.ones((self.batch_size, self.resp_dur, self.nneuron))  # resp
        # print(L1.shape)

        R1 = np.random.poisson(L1)
        Rd = np.random.poisson(Ld)
        Rr = np.random.poisson(Lr)
        print(G.shape)
        print(G[0][1])
        print(L1[0][1])
        print("R1.shape: ", R1.shape)

        example_input = np.concatenate((R1, Rd, Rr), axis=1)
        example_output = np.repeat(S[:, np.newaxis, :], self.total_dur, axis=1)

        print("example_output.shape: ", example_output.shape)

        cum_R1 = np.sum(R1, axis=1)
        print(cum_R1.shape)
        mu_x = np.asarray([np.arctan2(np.dot(cum_R1[:, i * self.n_in:(i + 1) * self.n_in], np.sin(2.0 * self.phi)),
                                      np.dot(cum_R1[:, i * self.n_in:(i + 1) * self.n_in], np.cos(2.0 * self.phi))) for
                           i in range(self.n_loc)]) / 2.0
        mu_x = (mu_x > 0.0) * mu_x + (mu_x < 0.0) * (mu_x + np.pi)
        mu_x = mu_x.T

        return example_input, example_output, S, mu_x


def build_generators(ExptDict):
    # Unpack common variables
    n_loc = ExptDict["task"]["n_loc"]
    n_out = ExptDict["task"]["n_out"]
    tr_cond = ExptDict["tr_cond"]
    test_cond = ExptDict["test_cond"]
    n_in = ExptDict["n_in"]
    batch_size = ExptDict["batch_size"]
    stim_dur = ExptDict["stim_dur"]
    delay_dur = ExptDict["delay_dur"]
    resp_dur = ExptDict["resp_dur"]
    kappa = ExptDict["kappa"]
    spon_rate = ExptDict["spon_rate"]
    tr_max_iter = ExptDict["tr_max_iter"]
    test_max_iter = ExptDict["test_max_iter"]

    generator = DelayedEstimationTask(max_iter=tr_max_iter,
                                      batch_size=batch_size,
                                      n_loc=n_loc, n_in=n_in,
                                      n_out=n_out, stim_dur=stim_dur,
                                      delay_dur=delay_dur,
                                      resp_dur=resp_dur, kappa=kappa,
                                      spon_rate=spon_rate,
                                      tr_cond=tr_cond)

    test_generator = DelayedEstimationTask(max_iter=test_max_iter,
                                           batch_size=batch_size,
                                           n_loc=n_loc, n_in=n_in,
                                           n_out=n_out,
                                           stim_dur=stim_dur,
                                           delay_dur=delay_dur,
                                           resp_dur=resp_dur,
                                           kappa=kappa,
                                           spon_rate=spon_rate,
                                           tr_cond=test_cond)

    return generator, test_generator


diag_val = 0.98
offdiag_val = 0.0
wdecay_coeff = 0.0

# Models and model-specific parameters
model_list = [{"model_id": 'LeInitRecurrent', "diag_val": diag_val, "offdiag_val": offdiag_val},
              {"model_id": 'LeInitRecurrentWithFastWeights', "diag_val": diag_val, "offdiag_val": offdiag_val,
               "gamma": 0.0007},
              {"model_id": 'OrthoInitRecurrent', "init_val": diag_val},
              {"model_id": 'ResidualRecurrent', "leak_inp": 1.0, "leak_hid": 1.0},
              {"model_id": 'GRURecurrent', "diag_val": diag_val, "offdiag_val": offdiag_val},
              {"model_id": 'LeInitRecurrentWithLayerNorm', "diag_val": diag_val, "offdiag_val": offdiag_val},
              ]

# Tasks and task-specific parameters
task_list = [{"task_id": 'DE1', "n_out": 1, "n_loc": 1, "out_nonlin": lasagne.nonlinearities.linear},
             {"task_id": 'DE2', "n_out": 2, "n_loc": 2, "out_nonlin": lasagne.nonlinearities.linear},
             {"task_id": 'CD1', "n_out": 1, "n_loc": 1, "out_nonlin": lasagne.nonlinearities.sigmoid},
             {"task_id": 'CD2', "n_out": 1, "n_loc": 2, "out_nonlin": lasagne.nonlinearities.sigmoid},
             {"task_id": 'GDE2', "n_out": 1, "n_loc": 2, "out_nonlin": lasagne.nonlinearities.linear},
             {"task_id": 'VDE1', "n_out": 1, "n_loc": 1, "max_delay": 100, "out_nonlin": lasagne.nonlinearities.linear},
             {"task_id": 'Harvey2012', "n_out": 1, "sigtc": 15.0, "stim_rate": 1.0, "n_loc": 1,
              "out_nonlin": lasagne.nonlinearities.sigmoid},
             {"task_id": 'SINE', "n_out": 1, "n_loc": 1, "alpha": 0.25, "out_nonlin": lasagne.nonlinearities.linear},
             {"task_id": 'COMP', "n_out": 1, "n_loc": 1, "out_nonlin": lasagne.nonlinearities.sigmoid}
             ]

m_ind = 0
t_ind = 0

ExptDict = {"model": model_list[m_ind],
            "task": task_list[t_ind],
            "tr_cond": 'all_gains',
            "test_cond": 'all_gains',
            "n_hid": 500,
            "n_in": 50,
            "batch_size": 50,
            "stim_dur": 25,
            "delay_dur": 100,
            "resp_dur": 25,
            "kappa": 2.0,
            "spon_rate": 0.1,
            "tr_max_iter": 1,
            "test_max_iter": 1}

# Build task generators
generator, test_generator = build_generators(ExptDict)

for i, (example_input, example_output, s, opt) in generator:
    print(example_input[0][1])
    print(example_output.shape)
    print(s[0])
    print(opt[0])
    sns.heatmap(example_input[0])
    plt.show()



