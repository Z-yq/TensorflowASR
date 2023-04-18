from collections import UserDict
import librosa
import numpy as np
from scipy import signal
import rir_generator as rir
import random


class SignalSpecAug:
    def __init__(self, window=10, ratio=0.5) -> None:
        self.window = window
        self.ratio = ratio

    def augment(self, wav):
        stft = librosa.stft(wav, n_fft=1024, win_length=800, hop_length=160)
        h, w = stft.shape
        nums = int(w * self.ratio)
        ws = random.sample(list(range(w)), nums)
        hs = random.sample(list(range(h)), nums)
        for h_, w_ in zip(hs, ws):
            stft[
                max(h_ - self.window, 0) : h_ + self.window,
                max(w_ - self.window, 0) : w_ + self.window,
            ] *= 0.0
        inver_wav = librosa.istft(stft, win_length=800, hop_length=160)
        return inver_wav


class SignalVC:
    def __init__(self):
        from .tts_for_asr.vc_aug import VC_Aug

        self.vc_aug = VC_Aug()

    def augment(self, wav):
        spk = np.random.randint(0, 1882, 1)
        wav = self.vc_aug.convert(wav, spk)
        return wav


class SignalRIR:
    def __init__(self, sample_rate):
        self.sp = sample_rate

    def get_num(self, x, y, z):
        x_ = random.sample(list(range(x * 10)), 1)[0]
        y_ = random.sample(list(range(y * 10)), 1)[0]
        z_ = random.sample(list(range(z * 10)), 1)[0]
        return [x_ / 10.0, y_ / 10.0, z_ / 10.0]

    def augment(self, wav):
        wav = wav[:, np.newaxis]
        h = rir.generate(
            c=340,  # Sound velocity (m/s)
            fs=self.sp,  # Sample frequency (samples/s)
            r=self.get_num(5, 4, 6),
            s=self.get_num(5, 4, 6),  # Source position [x y z] (m)
            L=[5, 4, 6],  # Room dimensions [x y z] (m)
            reverberation_time=0.4,  # Reverberation time (s)
            nsample=4096,  # Number of output samples
        )

        # Convolve 2-channel signal with 3 impulse responses
        wav = signal.convolve(h[:, None, :], wav[:, :, None])

        wav = wav.mean(axis=-1)
        return wav.flatten()


class SignalMask:
    def __init__(self, zone=(0.1, 0.9), mask_ratio=0.3, mask_with_noise=True):
        self.zone = eval(zone)
        self.mask_ratio = mask_ratio
        self.mask_with_noise = mask_with_noise

    def augment(self, data):
        length = len(data)
        s = int(length * self.zone[0])
        e = int(length * self.zone[1])
        data_ = data[s:e]

        if self.mask_with_noise:
            mask_value = np.random.random(len(data_))
            mask = np.where(mask_value < self.mask_ratio, 0.0, 1.0)
            value = mask_value * np.where(mask == 0.0, 1.0, 0.0)
            data_ *= mask
            data_ += value

        else:
            mask_value = np.random.random(len(data_))
            mask = np.where(mask_value < self.mask_ratio, 0.0, 1.0)
            data_ *= mask
        data[s:e] = data_
        return data


class SignalNoise:
    def __init__(self, sample_rate=16000, SNR=[-10, 10], noises=""):
        with open(noises) as f:
            noises = f.readlines()
        noises = [i.strip() for i in noises]
        self.noises = noises
        self.SNR = SNR
        self.sample_rate = sample_rate

    def Add_noise(self, x, d, SNR):
        P_signal = np.sum(abs(x) ** 2)

        P_d = np.sum(abs(d) ** 2)

        P_noise = P_signal / 10 ** (SNR / 10)

        noise = np.sqrt(P_noise / P_d) * d
        num = len(x)
        if num < noise.shape[0]:
            pick_num = np.random.randint(0, noise.shape[0] - num)
            noise = noise[int(pick_num) : int(pick_num) + num]
        noise_signal = x[:num] + noise[:num]
        return noise_signal

    def augment(self, data):

        n_num = np.random.randint(0, len(self.noises))
        n_wav = librosa.load(self.noises[n_num], self.sample_rate)[0]
        while len(data) + 20 > len(n_wav):
            n_wav = np.hstack((n_wav, n_wav))
        start = np.random.randint(0, len(n_wav) - len(data) - 10)
        n_wav = n_wav[start : start + len(data)]
        SNR = np.random.randint(self.SNR[0], self.SNR[1])
        wav = self.Add_noise(data, n_wav, SNR)
        return wav


class SignalPitch:
    def __init__(self, zone=(0.2, 0.8), sample_rate=16000, factor=(-1, 5)):
        self.zone = eval(zone)
        self.factor = eval(factor)
        self.sr = sample_rate

    def augment(self, data):
        length = len(data)
        s = int(length * self.zone[0])
        e = int(length * self.zone[1])
        data_ = data[s:e]
        scale = self.factor[1] - self.factor[0]
        factor = np.random.random() * scale - scale / 2
        wav = librosa.effects.pitch_shift(y=data_, sr=self.sr, n_steps=factor)
        data[s:e] = wav
        return data


class SignalSpeed:
    def __init__(self, factor=(0.5, 2)):
        self.factor = eval(factor)

    def augment(self, data):
        factor = np.random.random() * self.factor[1]
        factor = np.clip(factor, self.factor[0], self.factor[1])
        wav = librosa.effects.time_stretch(y=data, rate=factor)
        return wav


class SignalHz:
    def augment(self, data):
        start = np.random.random()
        start = np.clip(start, 0.01, 0.699)
        b, a = signal.butter(3, [start, start + 0.3], "bandstop")  # 配置滤波器 8 表示滤波器的阶数
        filtedData = signal.filtfilt(b, a, data)  # data为要过滤的信号
        filtedData += np.random.random(filtedData.shape) * 0.001
        return filtedData


AUGMENTATIONS = {
    "noise": SignalNoise,
    "masking": SignalMask,
    "pitch": SignalPitch,
    "speed": SignalSpeed,
    "hz": SignalHz,
    "rir": SignalRIR,
    "vc": SignalVC,
    "spec_aug": SignalSpecAug,
}


class Augmentation(UserDict):
    def __init__(self, config: dict = None):
        self.parse(config)
        super(Augmentation, self).__init__(config)

    def __missing__(self, key):
        return None

    def available(self):
        return len(self.augmentations) > 0

    def parse(self, config):
        self.augmentations = []
        for key, value in config.items():
            au = AUGMENTATIONS.get(key, None)

            if au is None:
                raise KeyError(
                    f"No augmentation named: {key}\n"
                    f"Available augmentations: {AUGMENTATIONS.keys()}"
                )

            if value["active"]:
                value.pop("active")

                aug = au(**value)
                self.augmentations.append(aug)

    def process(self, wav):

        augmentation = random.sample(self.augmentations, 1)[0]
        data = augmentation.augment(wav)
        data = np.array(np.clip(data, -1.0, 1.0) * 32768, "int32") / 32768.0
        return data
