import logging
import random

import librosa
import numpy as np
import tensorflow as tf

from augmentations.augments import Augmentation
from utils.speech_featurizers import SpeechFeaturizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class VADDataLoader():
    def __init__(self, config_dict, training=True):
        self.speech_config = config_dict['speech_config']
        self.running_config = config_dict['running_config']
        self.augment_config = config_dict['augments_config']
        self.batch = config_dict['running_config']['batch_size']
        self.speech_featurizer = SpeechFeaturizer(self.speech_config)
        self.make_file_list(training)
        self.augment = Augmentation(self.augment_config)
        self.epochs = 1
        self.steps = 0

    def return_data_types(self):

        return (tf.float32, tf.float32, tf.float32)

    def return_data_shape(self):

        return (
            tf.TensorShape([self.batch, None, self.speech_config['frame_input']]),
            tf.TensorShape([self.batch, None, 1]),
            tf.TensorShape([self.batch, None, self.speech_config['frame_input']]),
        )

    def get_per_epoch_steps(self):
        return len(self.train_list) // self.batch

    def eval_per_epoch_steps(self):
        return len(self.test_list) // self.batch

    def make_file_list(self, training=True):
        train_list = self.speech_config['train_list']
        test_list = self.speech_config['eval_list']
        if training:
            with open(train_list, encoding='utf-8') as f:
                train_list = f.readlines()
            train_list = [i.strip() for i in train_list if i != '']

            self.train_list = train_list

            np.random.shuffle(self.train_list)
            with open(test_list, encoding='utf-8') as f:
                data = f.readlines()
            data = [i.strip() for i in data if i != '']
            self.test_list = data
            self.train_offset = 0
            self.test_offset = 0
            logging.info('load train list {} test list {}'.format(len(self.train_list), len(self.test_list)))
        else:
            with open(test_list, encoding='utf-8') as f:
                data = f.readlines()
            data = [i.strip() for i in data if i != '']
            self.test_list = data
            self.offset = 0

    def generate(self, train):
        x = []
        y2 = []
        y = []
        maxlen = self.speech_config['max_frames']
        for i in range(self.batch):
            samples = []
            if train:
                sn = random.sample([2, 3, 4, 5], 1)[0]

                for _ in range(sn):
                    line = self.train_list[self.train_offset]
                    self.train_offset += 1
                    if self.train_offset > len(self.train_list) - 1:
                        self.train_offset = 0
                        np.random.shuffle(self.train_list)
                        self.epochs += 1
                    samples.append(line)
            else:
                sn = random.sample([2, 3, 4, 5], 1)[0]
                for _ in range(sn):
                    line = self.test_list[self.test_offset]
                    self.test_offset += 1
                    if self.test_offset > len(self.test_list) - 1:
                        self.test_offset = 0
                    samples.append(line)

            wav = np.zeros(1)
            wav_target = np.zeros(1)
            label = np.zeros(1)
            for sample in samples:
                data = self.speech_featurizer.load_wav(sample)

                try:
                    to_cut = data / (np.abs(data).max() + 1e-6)
                except:
                    continue
                cuts = librosa.effects.split(to_cut, top_db=20, frame_length=800, hop_length=80)
                data_label = np.zeros_like(data)
                for s, e in cuts:
                    s = int(s)
                    e = int(e)
                    data_label[s:e] = 1.
                if np.random.random() < 0.45:
                    data = data / np.abs(data).max()
                    data *= (np.random.random() * 2. + 0.1)
                    data = np.clip(data, -1., 1.)
                if self.augment.available():
                    data = self.augment.process(data)

                wav = np.hstack((wav, np.zeros(3200), data))
                wav_target = np.hstack((wav_target, np.zeros(3200), to_cut))
                label = np.hstack((label, np.zeros(3200), data_label))

            if len(wav) > maxlen:
                start = np.random.randint(0, len(wav) - maxlen, 1)[0]
                wav = wav[start:start + maxlen]
                wav_target = wav_target[start:start + maxlen]
                label = label[start:start + maxlen]
            else:
                wav = np.hstack((np.random.random(8000) * 0.001, wav, np.random.random(maxlen) * 0.001))
                wav_target = np.hstack((np.random.random(8000) * 0.001, wav_target, np.random.random(maxlen) * 0.001))
                label = np.hstack((np.zeros(8000), label, np.zeros(maxlen)))
                wav = wav[:maxlen]
                wav_target = wav_target[:maxlen]
                label = label[:maxlen]

            train_wav = wav.reshape([-1, self.speech_config['frame_input']])
            wav_target = wav_target.reshape([-1, self.speech_config['frame_input']])
            label = label.reshape([-1, self.speech_config['frame_input']])
            label = np.mean(label, -1, keepdims=True)
            label = np.where(label > self.speech_config['voice_thread'], 1., 0.)
            x.append(train_wav)
            y.append(label)
            y2.append(wav_target)
        x = np.array(x, 'float32')
        y = np.array(y, 'float32')
        y2 = np.array(y2, 'float32')

        return x, y, y2

    def generator(self, train=True):
        while 1:
            x, y, y2 = self.generate(train)
            if x.shape[0] == 0:
                logging.info('load data length zero,continue')
                continue

            yield x, y, y2