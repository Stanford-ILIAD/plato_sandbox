import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file_pattern', type=str, default="ft_%d.npy")
parser.add_argument('--eps', type=int, nargs='+', default=[-1])
parser.add_argument('--use_ep_range', action='store_true')  # excludes the end
args = parser.parse_args()
parser.add_argument('--legend', action='store_true')
parser.add_argument('--forces_only', action='store_true',
                    help="Use forces only (otherwise uses all forces and torques")


if len(args.eps) == 1:
    if args.eps[0] == -1:
        # all
        raise NotImplementedError
    else:
        # 0 ... max
        eps = list(range(args.eps[0]))

else:
    eps = args.eps
    if args.use_ep_range:
        assert len(eps) == 2
        eps = list(range(eps[0], eps[1]))

# file path
path = os.path.abspath(args.file_pattern)
pp = os.path.dirname(path)
assert os.path.exists(pp), pp

all_ft_seqs = []

for e in eps:
    # pattern must contain this
    fname = path % e
    force_torques = np.load(fname)
    all_ft_seqs.append(force_torques)

# E lists (Ni, 3), for each
forces = [fseq[..., :3] for fseq in all_ft_seqs]
torques = [fseq[..., 3:] for fseq in all_ft_seqs]

fig = plt.figure(figsize=(10,6), tight_layout=True)
axes = fig.subplots(nrows=2, ncols=3)

labels = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
for r in range(2):
    for c in range(3):
        ax = axes[r][c]
        ax.set_title(labels[r * 3 + c])
        for seq in all_ft_seqs:
            print(seq.shape)
            ax.plot(range(seq.shape[0]), seq[:, r * 3 + c])

plt.show()