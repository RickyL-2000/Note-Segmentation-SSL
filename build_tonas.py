# %%
import os

import numpy as np
from tqdm import tqdm
import pretty_midi
import wave

from src.utils.myutils import *

np.random.seed(42)

testset_ratio = 0.2

hop_length = 320
sr = 16000

# %%
f_names = []
trainset = []
testset = []

def find_files():
    global f_names
    with open(f"./meta/tonas.txt") as f:
        for f_name in f.readlines():
            f_names.append(f_name.strip())
    f_names.sort()

def split_dataset():
    np.random.seed(42)
    global trainset, testset
    testset_size = int(len(f_names) * testset_ratio)
    _testset = set(np.random.choice(f_names, testset_size, replace=False))
    for f_name in tqdm(f_names, desc="generating train and test paths"):
        if f_name in _testset:
            testset.append(f_name)
        else:
            trainset.append(f_name)
    trainset.sort()
    testset.sort()

find_files()
split_dataset()

# %%
np.random.seed(42)

def gen_meta_file():
    global trainset, testset
    with open(f"./meta/TONAS_train.txt", "w", encoding="utf-8") as f:
        for i, f_name in enumerate(trainset):
            f.write(f_name + ("\n" if i != len(trainset) - 1 else ""))
    with open(f"./meta/TONAS_test.txt", "w", encoding="utf-8") as f:
        for i, f_name in enumerate(testset):
            f.write(f_name + ("\n" if i != len(testset) - 1 else ""))

gen_meta_file()

# %%
def build_dataset():
    global trainset, testset
    mkdir(["./data/tonas_train/", "./data/tonas_train/wav/", "./data/tonas_train/sdt/", "./data/tonas_train/pitch/",
           "./data/tonas_train/onoffset_intervals/", "./data/tonas_train/pitch_intervals/"])
    mkdir(["./data/tonas_test/", "./data/tonas_test/wav/", "./data/tonas_test/sdt/", "./data/tonas_test/pitch/",
           "./data/tonas_test/onoffset_intervals/", "./data/tonas_test/pitch_intervals/"])

    def _build_dataset(dataset, dataset_path):
        for idx, f_name in tqdm(enumerate(dataset), total=len(dataset), ncols=80, desc=f"generating {dataset_path}"):
            # 复制wav文件
            wav_path = f"./data/TONAS/wav/{f_name}.wav"
            exe_cmd(f'cp "{wav_path}" {dataset_path}/wav/')

            # 读取并复制sdt文件
            sdt_path = f"./data/TONAS/sdt/{f_name}_sdt.npy"
            sdt = np.load(sdt_path, allow_pickle=True)
            exe_cmd(f'cp "{sdt_path}" {dataset_path}/sdt/')

            onoffset_intervals = []
            pitches = np.zeros(sdt.shape[0])
            pitch_intervals = []
            time_step = hop_length / sr
            with open(f"./data/TONAS/labels/{f_name}.notes.Corrected") as f:
                _ = f.readline()
                for line in f.readlines():
                    if line.strip() == "":
                        continue
                    line = line.strip().split(",")
                    onset, duration, pitch, energy = [float(line[i].strip()) for i in range(len(line))]

                    # 生成 onoffset_intervals
                    onoffset_intervals.append([onset, onset+duration])

                    # 生成 pitch 和 pitch_intervals
                    note_start = int(onset / time_step)
                    note_end = int((onset + duration) / time_step)
                    pitches[note_start: note_end] = note2freq(pitch)
                    pitch_intervals.append(note2freq(pitch))
            np.save(f"{dataset_path}/onoffset_intervals/{f_name}_oi.npy", onoffset_intervals)
            np.save(f"{dataset_path}/pitch/{f_name}_pitch.npy", pitches)
            np.save(f"{dataset_path}/pitch_intervals/{f_name}_pi.npy", pitch_intervals)

    _build_dataset(trainset, "./data/tonas_train")
    _build_dataset(testset, "./data/tonas_test")

build_dataset()
