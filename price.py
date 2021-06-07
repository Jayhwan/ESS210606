import numpy as np
day = 1
p = np.random.uniform(0, 5, (10, 24*day, 2))
ps = np.minimum(p[:, :, 0], p[:, :, 1])
pb = np.maximum(p[:, :, 0], p[:, :, 1])

np.save("ps.npy", ps)
np.save("pb.npy", pb)

ps = np.load("ps.npy", allow_pickle=True)
pb = np.load("pb.npy", allow_pickle=True)
print(ps[0, :])
print(pb[0, :])