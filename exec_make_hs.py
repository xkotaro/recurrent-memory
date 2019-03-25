import numpy as np

import torch

import make_hierarchical_signals
import matplotlib.pyplot as plt
import seaborn as sns

signals = []
targets = []
batch_size = 3
for i in range(batch_size):
    signal, target = make_hierarchical_signals.hierarchical_signals(100, spon_rate=0.01)
    signals.append(signal)
    targets.append(target)

signals = np.array(signals)
targets = np.array(targets)

signals = torch.from_numpy(signals)
targets = torch.from_numpy(targets)

batched_signals = signals[:, :75, :]
batched_targets = targets[:, :75, :]

print(type(signals), signals.shape)
print(type(targets), targets.shape)

print(type(batched_signals), batched_signals.shape)
print(type(batched_targets), batched_targets.shape)




