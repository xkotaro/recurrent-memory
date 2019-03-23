import numpy as np

import torch

import make_hierarchical_signals
import matplotlib.pyplot as plt
import seaborn as sns

signals = []
targets = []
batch_size = 50
for i in range(batch_size):
    signal, target = make_hierarchical_signals.hierarchical_signals(5000, spon_rate=0.01)
    signals.append(signal)
    targets.append(target)

signals = np.array(signals)
targets = np.array(targets)

signals = torch.from_numpy(signals)
targets = torch.from_numpy(targets)

print(type(signals), signals.shape)
print(type(targets), targets.shape)


