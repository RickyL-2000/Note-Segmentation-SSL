import os
import subprocess
import numpy as np

def exe_cmd(cmd, verbose=True):
    """
    :return: (stdout, stderr=None)
    """
    r = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ret = r.communicate()
    r.stdout.close()
    if verbose:
        res = str(ret[0].decode()).strip()
        if res:
            print(res)
    if ret[1] is not None:
        print(str(ret[0].decode()).strip())
    return ret

def mkdir(path):
    if type(path) == str:
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        return path
    elif type(path) == list:
        for p in path:
            mkdir(p)

def audio_len(path):
    """
    :return: 返回 microseconds
    """

    def _time2sec(time_str):
        print(time_str)
        start = 0
        end = time_str.find(':')
        ret = int(time_str[start: end]) * 3600
        start = end + 1
        end = time_str.find(':', start)
        ret += int(time_str[start: end]) * 60
        start = end + 1
        ret = float(time_str[start:]) + float(ret)
        return ret

    info = exe_cmd("ffprobe " + path, verbose=False)
    # pattern = re.compile("Duration: (.*?):(.*?):(.*?), start")
    # matcher = pattern.match(info[0].decode())
    text = info[0].decode()
    time_str = text[text.find("Duration") + 10: text.find(", start")]
    length = _time2sec(time_str)
    return length * 1000

def note2freq(note, a4=440):
    if type(note) is np.ndarray:
        ret = np.zeros(shape=note.shape)
        tmp = a4 * 2 ** ((note - 69) / 12)
        ret[~np.isclose(note, -1)] = tmp[~np.isclose(note, -1)]
        return ret
    if np.isclose(note, -1):
        return 0.0
    return a4 * 2 ** ((note - 69) / 12)

def freq2note(freq, a4=440):
    if type(freq) is np.ndarray:
        ret = -np.ones(shape=freq.shape)
        tmp = np.log2(freq/a4 + 1e-6) * 12 + 69
        ret[~np.isclose(freq, 0.0)] = tmp[~np.isclose(freq, 0.0)]
        return ret
    if np.isclose(freq, 0.0):
        return -1
    return np.log2(freq / a4) * 12 + 69
