# %%
import os

import numpy as np
from tqdm import tqdm
import pretty_midi
import wave

from src.utils.myutils import *

np.random.seed(42)

testset_ratio = 0.2
small_dataset_ratio = 0.3
mini_dataset_ratio = 0.05
tiny_dataset_ratio = 0.01

hop_length = 320
sr = 16000

# %%
short_paths = []    # 短句
long_paths = []     # 整首歌
train_paths = []
test_paths = []
trainset = []
testset = []

def find_files():
    global short_paths
    for song_name in os.listdir(f"./data/lasinger-short3/wav/"):
        long_paths.append(f"./data/lasinger-short3/wav/{song_name}")
        for sentence_name in os.listdir(f"./data/lasinger-short3/wav/{song_name}"):
            short_paths.append(f"./data/lasinger-short3/wav/{song_name}/{sentence_name}")
    # short_paths.sort()

def split_dataset():
    np.random.seed(42)
    global short_paths, long_paths, train_paths, test_paths
    testset_size = int(len(long_paths) * testset_ratio)
    _testset = set(np.random.choice(long_paths, testset_size, replace=False))
    for long_path in tqdm(long_paths, desc="generating train and test paths"):
        if long_path in _testset:
            for sentence_name in os.listdir(long_path):
                testset.append(sentence_name[:-4])
                test_paths.append(f"{long_path}/{sentence_name}")
        else:
            for sentence_name in os.listdir(long_path):
                trainset.append(sentence_name[:-4])
                train_paths.append(f"{long_path}/{sentence_name}")
    train_paths.sort()
    test_paths.sort()
    trainset.sort()
    testset.sort()

find_files()
split_dataset()

# %%
np.random.seed(42)

def gen_meta_file():
    global train_paths, test_paths
    with open(f"./meta/lasinger_train.txt", "w", encoding="utf-8") as f:
        for i, train_path in enumerate(train_paths):
            f.write(os.path.basename(train_path)[:-4] + ("\n" if i != len(train_paths) - 1 else ""))
    with open(f"./meta/lasinger_test.txt", "w", encoding="utf-8") as f:
        for i, test_path in enumerate(test_paths):
            f.write(os.path.basename(test_path)[:-4] + ("\n" if i != len(test_paths) - 1 else ""))

def gen_small_meta_file():
    global train_paths, test_paths
    small_train_paths = np.random.choice(train_paths, int(len(train_paths) * small_dataset_ratio), replace=False)
    small_test_paths = np.random.choice(test_paths, int(len(test_paths) * small_dataset_ratio), replace=False)
    with open(f"./meta/lasinger_train_small.txt", "w", encoding="utf-8") as f:
        for i, train_path in enumerate(small_train_paths):
            f.write(os.path.basename(train_path)[:-4] + ("\n" if i != len(small_train_paths) - 1 else ""))
    with open(f"./meta/lasinger_test_small.txt", "w", encoding="utf-8") as f:
        for i, test_path in enumerate(small_test_paths):
            f.write(os.path.basename(test_path)[:-4] + ("\n" if i != len(small_test_paths) - 1 else ""))

mini_train_paths = []
mini_test_paths = []
def gen_mini_meta_file():
    global train_paths, test_paths, mini_train_paths, mini_test_paths
    mini_train_paths = np.random.choice(train_paths, int(len(train_paths) * mini_dataset_ratio), replace=False)
    mini_test_paths = np.random.choice(test_paths, int(len(test_paths) * mini_dataset_ratio), replace=False)
    # with open(f"./meta/lasinger_train_mini.txt", "w", encoding="utf-8") as f:
    #     for i, train_path in enumerate(mini_train_paths):
    #         f.write(os.path.basename(train_path)[:-4] + ("\n" if i != len(mini_train_paths) - 1 else ""))
    # with open(f"./meta/lasinger_test_mini.txt", "w", encoding="utf-8") as f:
    #     for i, test_path in enumerate(mini_test_paths):
    #         f.write(os.path.basename(test_path)[:-4] + ("\n" if i != len(mini_test_paths) - 1 else ""))

def gen_tiny_meta_file():
    global train_paths, test_paths
    tiny_train_paths = np.random.choice(train_paths, int(len(train_paths) * tiny_dataset_ratio), replace=False)
    tiny_test_paths = np.random.choice(test_paths, int(len(test_paths) * tiny_dataset_ratio), replace=False)
    with open(f"./meta/lasinger_train_tiny.txt", "w", encoding="utf-8") as f:
        for i, train_path in enumerate(tiny_train_paths):
            f.write(os.path.basename(train_path)[:-4] + ("\n" if i != len(tiny_train_paths) - 1 else ""))
    with open(f"./meta/lasinger_test_tiny.txt", "w", encoding="utf-8") as f:
        for i, test_path in enumerate(tiny_test_paths):
            f.write(os.path.basename(test_path)[:-4] + ("\n" if i != len(tiny_test_paths) - 1 else ""))

def gen_voice_type_meta_file():
    # from mini set
    global train_paths, test_paths, mini_test_paths
    """
    type 0: 男低音
    type 1: 声乐男+普男
    type 2: 声乐女+普女
    type 3: 女高音
    """
    with open(f"./meta/lasinger_test_type0.txt", "w", encoding="utf-8") as f:
        for i, test_path in enumerate(mini_test_paths):
            if "男低音" in test_path:
                f.write(os.path.basename(test_path)[:-4] + ("\n" if i != len(mini_test_paths) - 1 else ""))
    with open(f"./meta/lasinger_test_type1.txt", "w", encoding="utf-8") as f:
        for i, test_path in enumerate(mini_test_paths):
            if "声乐男声" in test_path or "普通男声" in test_path:
                f.write(os.path.basename(test_path)[:-4] + ("\n" if i != len(mini_test_paths) - 1 else ""))
    with open(f"./meta/lasinger_test_type2.txt", "w", encoding="utf-8") as f:
        for i, test_path in enumerate(mini_test_paths):
            if ("声乐" in test_path and "女" in test_path) or "女_" in test_path:
                f.write(os.path.basename(test_path)[:-4] + ("\n" if i != len(mini_test_paths) - 1 else ""))
    with open(f"./meta/lasinger_test_type3.txt", "w", encoding="utf-8") as f:
        for i, test_path in enumerate(mini_test_paths):
            if "女高音" in test_path:
                f.write(os.path.basename(test_path)[:-4] + ("\n" if i != len(mini_test_paths) - 1 else ""))


# gen_meta_file()
# gen_small_meta_file()
gen_mini_meta_file()
# gen_tiny_meta_file()
gen_voice_type_meta_file()

# %%
def build_dataset():
    global trainset, testset
    mkdir(["./data/lasinger_train/", "./data/lasinger_train/wav/", "./data/lasinger_train/sdt/",
           "./data/lasinger_train/pitch/", "./data/lasinger_train/onoffset_intervals",
           "./data/lasinger_train/pitch_intervals"])
    mkdir(["./data/lasinger_test/", "./data/lasinger_test/wav/", "./data/lasinger_test/sdt/",
           "./data/lasinger_test/pitch/", "./data/lasinger_test/onoffset_intervals",
           "./data/lasinger_test/pitch_intervals"])

    def _build_dataset(dataset, dataset_path):
        for idx, sentence_name in tqdm(enumerate(dataset), total=len(dataset), ncols=80, desc=f"generating {dataset_path}"):
            # 复制文件
            singer, song_name, sentence_idx = sentence_name.split("#")
            wav_path = f"./data/lasinger-short3/wav/{singer}#{song_name}/{sentence_name}.wav"
            exe_cmd(f'cp "{wav_path}" {dataset_path}/wav/')

            # 生成 sdt 文件
            wav_f = wave.open(wav_path)
            wav_len = wav_f.getnframes() / float(wav_f.getframerate())
            time_step = hop_length / sr
            sdt = np.zeros((int(wav_len * (sr / hop_length)), 6), dtype=int)
            sdt[:, [0, 2, 4]] = 1
            mid = pretty_midi.PrettyMIDI(f"./data/lasinger-short3/midi/{singer}#{song_name}/{sentence_name}.mid")
            for note_idx, note in enumerate(mid.instruments[0].notes):
                note_start = int(note.start / time_step)
                note_end = int(note.end / time_step)
                sdt[note_start: note_end, 0] = 0
                sdt[note_start: note_end, 1] = 1
                sdt[max(0, note_start - 2): note_start + 3, 2] = 0
                sdt[max(0, note_start - 2): note_start + 3, 3] = 1
                sdt[max(0, note_end - 3): note_end + 2, 4] = 0
                sdt[max(0, note_end - 3): note_end + 2, 5] = 1
            np.save(f"{dataset_path}/sdt/{sentence_name}_sdt.npy", sdt)

            # 生成 onoffset_intervals
            onoffset_intervals = []
            for note_idx, note in enumerate(mid.instruments[0].notes):
                onoffset_intervals.append([note.start, note.end])
            onoffset_intervals = np.array(onoffset_intervals)
            np.save(f"{dataset_path}/onoffset_intervals/{sentence_name}_oi.npy", onoffset_intervals)

            # 生成 pitch 和 pitch_intervals
            pitch = np.zeros(int(wav_len * (sr / hop_length)))
            pitch_intervals = []
            for note_idx, note in enumerate(mid.instruments[0].notes):
                note_start = int(note.start / time_step)
                note_end = int(note.end / time_step)
                pitch[note_start: note_end] = note2freq(note.pitch)
                pitch_intervals.append(note2freq(note.pitch))
            pitch_intervals = np.array(pitch_intervals)
            np.save(f"{dataset_path}/pitch/{sentence_name}_pitch.npy", pitch)
            np.save(f"{dataset_path}/pitch_intervals/{sentence_name}_pi.npy", pitch_intervals)

    _build_dataset(trainset, "./data/lasinger_train")
    _build_dataset(testset, "./data/lasinger_test")

build_dataset()
