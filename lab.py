# %%
import numpy as np

# %%
sdt = np.load("data/ISMIR2014/sdt/afemale1_sdt.npy")
onoffset_intervals = np.load("data/ISMIR2014/onoffset_intervals/afemale1_oi.npy")
pitch = np.load("data/ISMIR2014/pitch/afemale1_pitch.npy")
pitch_intervals = np.load("data/ISMIR2014/pitch_intervals/afemale1_pi.npy")

"""
经过观察：
pitch在 index = 50 的时候非零，而对应的在onoffset_intervals里是1.01秒，所以sample rate = 50，time_step = 0.02s

果然，在代码中我找到：
    fs = 16000.0 # sampling frequency
    Hop = 320 # hop size (in sample)
在这种情况下，feature的sample rate = 16000/320 = 50

此外，我还发现onset和offset的变化的地方，都是取了5个采样点。即以onset或者offset的时间点周围的5个采样点作为tolerance
"""

# %% 再测试
sdt = np.load("data/lasinger_train/sdt/男低音1#爱情转移#0_sdt.npy")
onoffset_intervals = np.load("data/lasinger_train/onoffset_intervals/男低音1#爱情转移#0_oi.npy")
pitch = np.load("data/lasinger_train/pitch/男低音1#爱情转移#0_pitch.npy")
pitch_intervals = np.load("data/lasinger_train/pitch_intervals/男低音1#爱情转移#0_pi.npy")
