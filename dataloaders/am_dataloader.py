import os

import numpy as np
import pypinyin
import tensorflow as tf
import logging

from augmentations.augments import Augmentation
from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class AM_DataLoader():

    def __init__(self, config_dict, training=True):
        self.speech_config = config_dict['speech_config']

        self.text_config = config_dict['decoder_config']
        self.augment_config = config_dict['augments_config']
        self.streaming=self.speech_config['streaming']
        self.chunk=self.speech_config['sample_rate']*self.speech_config['streaming_bucket']
        self.batch = config_dict['learning_config']['running_config']['batch_size']
        self.speech_featurizer = SpeechFeaturizer(self.speech_config)
        self.text_featurizer = TextFeaturizer(self.text_config)
        self.make_file_list(self.speech_config['train_list'] if training else self.speech_config['eval_list'], training)
        self.augment = Augmentation(self.augment_config)
        self.init_text_to_vocab()
        self.epochs = 1
        self.LAS = False
        self.steps = 0

    def load_state(self, outdir):
        try:

            dg_state = np.load(os.path.join(outdir, 'dg_state.npz'))

            self.epochs = int(dg_state['epoch'])
            self.train_offset = int(dg_state['train_offset'])
            train_list = dg_state['train_list'].tolist()
            if len(train_list) != len(self.train_list):
                logging.info('history train list not equal new load train list ,data loader use init state')
                self.epochs = 0
                self.train_offset = 0
        except FileNotFoundError:
            logging.info('not found state file,init state')
        except:
            logging.info('load state falied,use init state')

    def save_state(self, outdir):

        np.savez(os.path.join(outdir, 'dg_state.npz'), epoch=self.epochs, train_offset=self.train_offset,
                 train_list=self.train_list)

    def return_data_types(self):
        if self.LAS:
            return ( tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)
        else:
            return (tf.float32, tf.int32, tf.int32, tf.int32)

    def return_data_shape(self):
        f, c = self.speech_featurizer.compute_feature_dim()
        if self.LAS:
            return (
                tf.TensorShape([None, None, 1]) if self.speech_config['use_mel_layer'] else tf.TensorShape(
                    [None, None, f, c]),

                tf.TensorShape([None, ]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, ]),
                tf.TensorShape([None, None, None])
            )
        else:
            return (
                tf.TensorShape([None, None, 1]) if self.speech_config['use_mel_layer'] else tf.TensorShape(
                    [None, None, f, c]),

                tf.TensorShape([None, ]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, ])
            )

    def get_per_epoch_steps(self):
        return len(self.train_list) // self.batch

    def eval_per_epoch_steps(self):
        return len(self.test_list) // self.batch

    def init_text_to_vocab(self):
        pypinyin.load_phrases_dict({'调大': [['tiáo'], ['dà']],
                                    '调小': [['tiáo'], ['xiǎo']],
                                    '调亮': [['tiáo'], ['liàng']],
                                    '调暗': [['tiáo'], ['àn']],
                                    '肖': [['xiāo']],
                                    '英雄传': [['yīng'], ['xióng'], ['zhuàn']],
                                    '新传': [['xīn'], ['zhuàn']],
                                    '外传': [['wài'], ['zhuàn']],
                                    '正传': [['zhèng'], ['zhuàn']], '水浒传': [['shuǐ'], ['hǔ'], ['zhuàn']]
                                    })

        def text_to_vocab_func(txt):
            pins = pypinyin.pinyin(txt)
            pins = [i[0] for i in pins]
            return pins

        self.text_to_vocab = text_to_vocab_func



    def make_file_list(self, wav_list, training=True):
        with open(wav_list, encoding='utf-8') as f:
            data = f.readlines()
        data = [i.strip() for i in data if i != '']
        num = len(data)
        if training:
            self.train_list = data[:int(num * 0.99)]
            self.test_list = data[int(num * 0.99):]
            np.random.shuffle(self.train_list)
            self.train_offset = 0
            self.test_offset = 0
            logging.info('load train list {} test list{}'.format(len(self.train_list),len(self.test_list)))
        else:
            self.test_list = data
            self.offset = 0

    def only_chinese(self, word):
        txt = ''
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                txt += ch
            else:
                continue

        return txt

    def eval_data_generator(self):
        sample = self.test_list[self.offset:self.offset + self.batch]
        self.offset += self.batch
        speech_features = []
        input_length = []
        y1 = []
        label_length1 = []
        max_input = 0
        max_label1 = 0
        for i in sample:
            wp, txt = i.strip().split('\t')
            txt = txt.replace(' ', '')
            try:
                data = self.speech_featurizer.load_wav(wp)
            except:
                logging.info('{} load data failed,skip'.format(wp))
                continue
            if len(data) < 400:
                logging.info('{} wav too short < 25ms,skip'.format(wp))
                continue
            elif len(data) > self.speech_featurizer.sample_rate * self.speech_config['wav_max_duration']:
                logging.info(
                    '{} duration out of wav_max_duration({}) ,skip'.format(wp, self.speech_config['wav_max_duration']))
                continue
            if self.speech_config['only_chinese']:
                txt = self.only_chinese(txt)
            if self.speech_config['use_mel_layer']:
                if not self.streaming:
                    speech_feature = data / np.abs(data).max()
                    speech_feature = np.expand_dims(speech_feature, -1)
                    in_len = len(speech_feature) // (
                            self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) *
                            self.speech_config['stride_ms'])
                else:
                    speech_feature = data
                    speech_feature = np.expand_dims(speech_feature, -1)
                    reduce=self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) *self.speech_config['stride_ms']
                    in_len = len(speech_feature) //self.chunk
                    if len(speech_feature) %self.chunk!=0:
                        in_len+=1
                    chunk_times=self.chunk//reduce
                    if self.chunk%reduce!=0:
                        chunk_times+=1
                    in_len*=chunk_times
                
            else:
                speech_feature = self.speech_featurizer.extract(data)
                in_len = int(speech_feature.shape[0] // self.speech_config['reduction_factor'])
            max_input = max(max_input, speech_feature.shape[0])

            py = self.text_to_vocab(txt)
            if  self.check_valid(py, self.text_featurizer.vocab_array) is not True:
                logging.info(' {} txt pinyin {} not all in tokens,skip'.format(txt, self.check_valid(py, self.text_featurizer.vocab_array)))
                continue
            text_feature = self.text_featurizer.extract(py)

            if in_len < len(text_feature):
                logging.info('{} feature length < pinyin length,skip'.format(wp))
                continue
            max_input = max(max_input, len(speech_feature))
            max_label1 = max(max_label1, len(text_feature))
            speech_features.append(speech_feature)
            input_length.append(in_len)
            y1.append(np.array(text_feature))
            label_length1.append(len(text_feature))

        if self.speech_config['use_mel_layer']:
            if self.streaming:
                max_input=max_input//self.chunk*self.chunk+self.chunk
            speech_features = self.speech_featurizer.pad_signal(speech_features, max_input)

        else:
            for i in range(len(speech_features)):

                if speech_features[i].shape[0] < max_input:
                    pad = np.ones([max_input - speech_features[i].shape[0], speech_features[i].shape[1],
                                   speech_features[i].shape[2]]) * speech_features[i].min()
                    speech_features[i] = np.vstack((speech_features[i], pad))

        for i in range(len(y1)):
            if y1[i].shape[0] < max_label1:
                pad = np.ones(max_label1 - y1[i].shape[0]) * self.text_featurizer.pad
                y1[i] = np.hstack((y1[i], pad))

        x = np.array(speech_features, 'float32')
        y1 = np.array(y1, 'int32')

        input_length = np.array(input_length, 'int32')
        label_length1 = np.array(label_length1, 'int32')

        return x, input_length, y1, label_length1

    def check_valid(self, txt, vocab_list):
        if len(txt) == 0:
            return False
        for n in txt:
            if n in vocab_list:
                pass
            else:
                return n
        return True

    def GuidedAttentionMatrix(self, N, T, g=0.2):
        W = np.zeros((N, T), dtype=np.float32)
        for n in range(N):
            for t in range(T):
                W[n, t] = 1 - np.exp(-(t / float(T) - n / float(N)) ** 2 / (2 * g * g))
        return W

    def guided_attention(self, input_length, targets_length, inputs_shape, mel_target_shape):
        att_targets = []
        for i, j in zip(input_length, targets_length):
            i = int(i)
            step = int(j)
            pad = np.ones([inputs_shape, mel_target_shape]) * -1.
            pad[i:, :step] = 1
            att_target = self.GuidedAttentionMatrix(i, step, 0.2)
            pad[:att_target.shape[0], :att_target.shape[1]] = att_target
            att_targets.append(pad)
        att_targets = np.array(att_targets)

        return att_targets.astype('float32')

    def generate(self, train=True):

        sample = []
        speech_features = []
        input_length = []
        y1 = []
        label_length1 = []

        max_input = 0
        max_label1 = 0
        if train:
            batch = self.batch // 2 if self.augment.available() else self.batch
        else:
            batch = self.batch

        for i in range(batch * 10):
            if train:
                line = self.train_list[self.train_offset]
                self.train_offset += 1
                if self.train_offset > len(self.train_list) - 1:
                    self.train_offset = 0
                    np.random.shuffle(self.train_list)
                    self.epochs += 1
            else:
                line = self.test_list[self.test_offset]
                self.test_offset += 1
                if self.test_offset > len(self.test_list) - 1:
                    self.test_offset = 0
            wp, txt = line.strip().split('\t')
            try:
                data = self.speech_featurizer.load_wav(wp)
            except:
                logging.info('{} load data failed,skip'.format(wp))
                continue
            if len(data) < 400:
                continue
            elif len(data) > self.speech_featurizer.sample_rate * self.speech_config['wav_max_duration']:
                logging.info('{} duration out of wav_max_duration({}),skip'.format(wp, self.speech_config['wav_max_duration']))
                continue
            if self.speech_config['only_chinese']:
                txt = self.only_chinese(txt)
            if self.speech_config['use_mel_layer']:
                if not self.streaming:
                    speech_feature = data / np.abs(data).max()
                    speech_feature = np.expand_dims(speech_feature, -1)
                    in_len = len(speech_feature) // (
                            self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) *
                            self.speech_config['stride_ms'])
                else:
                    speech_feature = data
                    speech_feature = np.expand_dims(speech_feature, -1)
                    reduce = self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) * \
                             self.speech_config['stride_ms']
                    in_len = len(speech_feature) // self.chunk
                    if len(speech_feature) % self.chunk != 0:
                        in_len += 1
                    chunk_times = self.chunk // reduce
                    if self.chunk % reduce != 0:
                        chunk_times += 1
                    in_len *= chunk_times
            else:
                speech_feature = self.speech_featurizer.extract(data)
                in_len = int(speech_feature.shape[0] // self.speech_config['reduction_factor'])

            py = self.text_to_vocab(txt)
            if self.check_valid(py, self.text_featurizer.vocab_array) is not True:
                logging.info(' {} txt pinyin {} not all in tokens,continue'.format(txt, self.check_valid(py, self.text_featurizer.vocab_array)))
                continue
            text_feature = self.text_featurizer.extract(py)

            if in_len < len(text_feature):
                logging.info('{} feature length < pinyin length,continue'.format(wp))
                continue
            max_input = max(max_input, len(speech_feature))
            max_label1 = max(max_label1, len(text_feature))
            speech_features.append(speech_feature)
            input_length.append(in_len)
            y1.append(np.array(text_feature))
            label_length1.append(len(text_feature))
            sample.append(line)
            if len(sample)==batch:
                break
        if train and self.augment.available():
            for i in sample:
                wp, txt = i.strip().split('\t')
                try:
                    data = self.speech_featurizer.load_wav(wp)
                except:
                    continue
                if len(data) < 400:
                    logging.info('{} wav too short < 25ms,skip'.format(wp))
                    continue
                elif len(data) > self.speech_featurizer.sample_rate * self.speech_config['wav_max_duration']:
                    continue
                data = self.augment.process(data)
                if self.speech_config['only_chinese']:
                    txt = self.only_chinese(txt)
                if self.speech_config['use_mel_layer']:
                    if not self.streaming:
                        speech_feature = data / np.abs(data).max()
                        speech_feature = np.expand_dims(speech_feature, -1)
                        in_len = len(speech_feature) // (
                                self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) *
                                self.speech_config['stride_ms'])
                    else:
                        speech_feature = data
                        speech_feature = np.expand_dims(speech_feature, -1)
                        reduce = self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) * \
                                 self.speech_config['stride_ms']
                        in_len = len(speech_feature) // self.chunk
                        if len(speech_feature) % self.chunk != 0:
                            in_len += 1
                        chunk_times = self.chunk // reduce
                        if self.chunk % reduce != 0:
                            chunk_times += 1
                        in_len *= chunk_times
                else:
                    speech_feature = self.speech_featurizer.extract(data)
                    in_len = int(speech_feature.shape[0] // self.speech_config['reduction_factor'])

                py = self.text_to_vocab(txt)
                if not self.check_valid(py, self.text_featurizer.vocab_array):
                    continue

                text_feature = self.text_featurizer.extract(py)

                if in_len < len(text_feature):
                    continue
                max_input = max(max_input, len(speech_feature))
                max_label1 = max(max_label1, len(text_feature))
                speech_features.append(speech_feature)

                input_length.append(in_len)
                y1.append(np.array(text_feature))
                label_length1.append(len(text_feature))

        if self.speech_config['use_mel_layer']:
            if self.streaming:
                reduce = self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) * \
                         self.speech_config['stride_ms']
                max_input = max_input // self.chunk * self.chunk + self.chunk
                max_in_len=max_input//self.chunk
                chunk_times = self.chunk // reduce
                if self.chunk % reduce != 0:
                    chunk_times += 1
                max_in_len*=chunk_times
                input_length=np.clip(input_length,0,max_in_len)
            speech_features = self.speech_featurizer.pad_signal(speech_features, max_input)

        else:
            for i in range(len(speech_features)):

                if speech_features[i].shape[0] < max_input:
                    pad = np.ones([max_input - speech_features[i].shape[0], speech_features[i].shape[1],
                                   speech_features[i].shape[2]]) * speech_features[i].min()
                    speech_features[i] = np.vstack((speech_features[i], pad))

        for i in range(len(y1)):
            if y1[i].shape[0] < max_label1:
                pad = np.ones(max_label1 - y1[i].shape[0]) * self.text_featurizer.pad
                y1[i] = np.hstack((y1[i], pad))

        x = np.array(speech_features, 'float32')
        y1 = np.array(y1, 'int32')

        input_length = np.array(input_length, 'int32')
        label_length1 = np.array(label_length1, 'int32')

        return x, input_length, y1, label_length1

    def generator(self, train=True):
        while 1:
            x, input_length, labels, label_length = self.generate(train)
            if x.shape[0] == 0:
                logging.info('load data length zero,continue')
                continue
            if self.LAS:
                guide_matrix = self.guided_attention(input_length, label_length, np.max(input_length),
                                                     label_length.max())
                yield x, input_length, labels, label_length, guide_matrix
            else:
                yield x, input_length, labels, label_length
