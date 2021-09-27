#!/usr/bin/env python3
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
from tqdm import trange
import sys

fn = sys.argv[1]
assert(fn[-4:] == '.txt')
outfn = fn[:-4]+'_filteredfast.txt'
data  = np.loadtxt(fn)

dt = 1/1000.

res = np.zeros_like(data)

my_filter = KalmanFilter(dim_x=2*data.shape[1], dim_z=data.shape[1])
my_filter.x = np.r_[data[0], data[0]*0]
my_filter.F = np.zeros((2*data.shape[1], 2*data.shape[1]))
my_filter.H = np.zeros((data.shape[1], 2*data.shape[1]))
# my_filter.R *= 5
my_filter.R *= 1
# my_filter.Q = Q_discrete_white_noise(2, dt, 0.8, block_size=data.shape[1])
my_filter.Q = Q_discrete_white_noise(2, dt, 3.0, block_size=data.shape[1])

for i in range(data.shape[1]):
    my_filter.F[2*i, 2*i] = 1.
    my_filter.F[2*i, 2*i+1] = 1.
    my_filter.F[2*i+1, 2*i] = 0
    my_filter.F[2*i+1, 2*i+1] = 1.

    my_filter.H[i, 2*i] = 1

res[0] = data[0]

for i in trange(1, len(data)):
    my_filter.predict()
    my_filter.update(data[i])

    x = my_filter.x
    # print(x)
    res[i] = x[::2]

np.savetxt(outfn, res)
plt.plot(data[:, -3])
plt.plot(res[:, -3])
plt.show()

