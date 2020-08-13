# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import os
from collections import UserDict
import random
import librosa
import numpy as np

class SignalMask():
    def __init__(self,
                 zone=(0.1, 0.9),

                 mask_ratio=0.3,

                 mask_with_noise=True):
        self.zone=eval(zone)
        self.mask_ratio=mask_ratio
        self.mask_with_noise=mask_with_noise
    def augment(self,data):
        length=len(data)
        s=int(length*self.zone[0])
        e=int(length*self.zone[1])
        data_=data[s:e]

        if self.mask_with_noise:
            mask_value = np.random.random(len(data_))
            mask = np.where(mask_value < self.mask_ratio, 0., 1.)
            value=mask_value*np.where(mask==0.,1.,0.)
            data_*=mask
            data_+=value

        else:
            mask_value = np.random.random(len(data_))
            mask = np.where(mask_value < self.mask_ratio, 0., 1.)
            data_*=mask
        data[s:e]=data_
        return data



class SignalNoise():
    def __init__(self,
                 sample_rate=16000,
                 SNR=[-10,10],

                 noises= None):
        if noises is not None:
            noises = glob.glob(os.path.join(noises, "**", "*.wav"), recursive=True)
            self.noises = [librosa.load(n, sr=sample_rate)[0] for n in noises]
        self.SNR=eval(SNR)

    def Add_noise(self, x, d, SNR):
        P_signal = np.sum(abs(x) ** 2)

        P_d = np.sum(abs(d) ** 2)

        P_noise = P_signal / 10 ** (SNR / 10)

        noise = np.sqrt(P_noise / P_d) * d
        num = min(x.shape[0], noise.shape[0])
        if num < noise.shape[0]:
            pick_num = np.random.randint(0, noise.shape[0] - num)
            noise = noise[int(pick_num):int(pick_num) + num]
        noise_signal = x[:num] + noise[:num]
        return noise_signal

    def augment(self, data):

        n_num = np.random.randint(0, len(self.noises))
        n_wav = self.noises[n_num]
        while len(data) + 20 > len(n_wav):
            n_wav = np.hstack((n_wav, n_wav))
        start = np.random.randint(0, len(n_wav) - len(data) - 10)
        n_wav = n_wav[start:start + len(data)]
        SNR = np.random.randint(self.SNR[0],self.SNR[1])
        wav = self.Add_noise(data, n_wav, SNR)
        return wav

class SignalPitch():
    def __init__(self,
                 zone=(0.2, 0.8),
                 sample_rate=16000,
                 factor=(-1, 5)):
        self.zone=eval(zone)
        self.factor=eval(factor)
        self.sr=sample_rate

    def augment(self,data):
        length = len(data)
        s = int(length * self.zone[0])
        e = int(length * self.zone[1])
        data_ = data[s:e]
        scale=self.factor[1]-self.factor[0]
        factor=np.random.random()*scale-scale/2
        wav=librosa.effects.pitch_shift(data_,self.sr,factor)
        data[s:e]=wav
        return data




class SignalSpeed():
    def __init__(self,
                 factor=(0.5, 2)):
        self.factor=eval(factor)
    def augment(self,data):
        factor=np.random.random()*self.factor[1]
        factor=np.clip(factor,self.factor[0],self.factor[1])
        wav=librosa.effects.time_stretch(data,factor)
        return wav








AUGMENTATIONS = {
    "noise": SignalNoise,
    "masking": SignalMask,
    "pitch": SignalPitch,
    "speed": SignalSpeed,
}


class Augmentation(UserDict):
    def __init__(self, config: dict = None):
        self.parse(config)
        super(Augmentation, self).__init__(config)

    def __missing__(self, key):
        return None
    def available(self):
        return len(self.augmentations)>0
    def parse(self,config):
        self.augmentations = []
        for key, value in config.items():
            au = AUGMENTATIONS.get(key, None)

            if au is None:
                raise KeyError(f"No augmentation named: {key}\n"
                               f"Available augmentations: {AUGMENTATIONS.keys()}")

            if value['active']:
                value.pop('active')

                aug = au(**value)
                self.augmentations.append(aug)

    def process(self,wav):

        augmentation=random.sample(self.augmentations,1)[0]
        data=augmentation.augment(wav)
        return data

