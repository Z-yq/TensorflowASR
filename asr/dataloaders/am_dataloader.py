import logging
import random

import numpy as np
import pypinyin
import tensorflow as tf

from augmentations.augments import Augmentation
from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import time

class AM_DataLoader():

    def __init__(self, config_dict, training=True):

        self.speech_config = config_dict['speech_config']
        self.phone_config = config_dict['inp_config']
        self.text_config = config_dict['tar_config']
        self.running_config=config_dict['running_config']
        self.augment_config = config_dict['augments_config']
        self.streaming = self.speech_config['streaming']
        self.chunk = self.speech_config['sample_rate'] * self.speech_config['streaming_bucket']
        self.batch = config_dict['running_config']['batch_size']
        self.speech_featurizer = SpeechFeaturizer(self.speech_config)
        self.phone_featurizer = TextFeaturizer(self.phone_config)
        self.text_featurizer = TextFeaturizer(self.text_config)
        self.make_file_list( training)
        self.augment = Augmentation(self.augment_config)
        self.init_text_to_vocab()
        self.epochs = 1
        self.steps = 0


    def return_data_types(self):

        return (tf.float32, tf.int32, tf.int32, tf.int32,tf.int32)

    def return_data_shape(self):

        return (
            tf.TensorShape([self.batch, None, 1]),

            tf.TensorShape([self.batch, ]),
            tf.TensorShape([self.batch, None]),
            tf.TensorShape([self.batch, ]),
            tf.TensorShape([self.batch, None]),
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
            phones = []
            for pin in pins:
                if pin in self.phone_featurizer.vocab_array:
                    phones += [pin]
                else:
                    phones += list(pin)
            # print(phones)
            return phones

        self.text_to_vocab = text_to_vocab_func

    def make_file_list(self, training=True):
        train_list=self.speech_config['train_list']
        test_list=self.speech_config['eval_list']
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
            self.test_offset = 0

    def only_chinese(self, word):
        txt = ''
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                txt += ch
            else:
                continue

        return txt

    def eval_data_generator(self):

        sample = []
        speech_features = []
        input_length = []
        phones = []
        phones_length = []
        txts = []
        max_input = 0
        batch = self.batch

        for i in range(batch * 10):

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
                logging.info(
                    '{} duration out of wav_max_duration({}),skip'.format(wp, self.speech_config['wav_max_duration']))
                continue
            if self.speech_config['only_chinese']:
                txt = self.only_chinese(txt)

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

            py = self.text_to_vocab(txt)
            if self.check_valid(py, self.phone_featurizer.vocab_array) is not True:
                logging.info(' {} txt phone {} not all in tokens,continue'.format(txt, self.check_valid(py,
                                                                                                        self.phone_featurizer.vocab_array)))
                continue
            if self.check_valid(txt, self.text_featurizer.vocab_array) is not True:
                logging.info(' {} txt phone {} not all in tokens,continue'.format(txt, self.check_valid(py,
                                                                                                        self.text_featurizer.vocab_array)))
                continue

            txt = list(txt)
            phone_feature = self.phone_featurizer.extract(py)
            text_feature = self.text_featurizer.extract(txt)+[self.text_featurizer.endid()]

            if in_len < len(phone_feature):
                logging.info('{} feature length < phone length,continue'.format(wp))
                continue
            max_input = max(max_input, len(speech_feature))

            speech_features.append(speech_feature)
            input_length.append(in_len)
            phones.append(np.array(phone_feature))
            txts.append(np.array(text_feature))
            phones_length.append(len(phone_feature))
            sample.append(line)
            if len(sample) == batch:
                break


        if self.streaming:
            max_input = max_input // self.chunk * self.chunk + self.chunk
        speech_features = self.speech_featurizer.pad_signal(speech_features, max_input)



        if self.streaming:
            reduce = self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) * \
                     self.speech_config['stride_ms']
            max_input = max_input // self.chunk * self.chunk + self.chunk
            max_in_len = max_input // self.chunk
            chunk_times = self.chunk // reduce
            if self.chunk % reduce != 0:
                chunk_times += 1
            max_in_len *= chunk_times
            input_length = np.clip(input_length, 0, max_in_len)
        speech_features = self.speech_featurizer.pad_signal(speech_features, max_input)

        phones = tf.keras.preprocessing.sequence.pad_sequences(phones, maxlen=max([len(i) for i in phones]),
                                                               padding='post', value=self.phone_featurizer.pad)
        txts = tf.keras.preprocessing.sequence.pad_sequences(txts, maxlen=max([len(i) for i in txts]), padding='post',
                                                             value=self.text_featurizer.pad)
        x = np.array(speech_features, 'float32')
        phones = np.array(phones, 'int32')
        txts = np.array(txts, 'int32')

        input_length = np.array(input_length, 'int32')
        phones_length = np.array(phones_length, 'int32')

        return x, input_length, phones, phones_length, txts

    def check_valid(self, txt, vocab_list):
        if len(txt) == 0:
            return False
        for n in txt:
            if n in vocab_list:
                pass
            else:
                return n
        return True


    def generate(self, train=True):

        sample = []
        speech_features = []
        input_length = []
        phones = []
        phones_length = []
        txts = []

        max_input = 0

        if train:
            batch = self.batch * 3 // 4 if self.augment.available() else self.batch
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
                logging.info(
                    '{} duration out of wav_max_duration({}),skip'.format(wp, self.speech_config['wav_max_duration']))
                continue
            if self.speech_config['only_chinese']:
                txt = self.only_chinese(txt)

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

            py = self.text_to_vocab(txt)
            if self.check_valid(py, self.phone_featurizer.vocab_array) is not True:
                logging.info(' {} txt phone {} not all in tokens,continue'.format(txt, self.check_valid(py,
                                                                                                        self.phone_featurizer.vocab_array)))
                continue
            if self.check_valid(txt, self.text_featurizer.vocab_array) is not True:
                logging.info(' {} txt  {} not all in tokens,continue'.format(txt, self.check_valid(txt,
                                                                                                        self.text_featurizer.vocab_array)))
                continue

            txt = list(txt)
            phone_feature = self.phone_featurizer.extract(py)
            text_feature = self.text_featurizer.extract(txt)+[self.text_featurizer.endid()]

            if in_len < len(phone_feature):
                logging.info('{} feature length < phone length,continue'.format(wp))
                continue
            max_input = max(max_input, len(speech_feature))

            speech_features.append(speech_feature)
            input_length.append(in_len)
            phones.append(np.array(phone_feature))
            txts.append(np.array(text_feature))
            phones_length.append(len(phone_feature))
            sample.append(line)
            if len(sample) == batch:
                break
        if train and self.augment.available():

            sample = random.sample(sample, self.batch // 4)
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

                py = self.text_to_vocab(txt)
                if self.check_valid(py, self.phone_featurizer.vocab_array) is not True:
                    logging.info(' {} txt phone {} not all in tokens,continue'.format(txt, self.check_valid(py,
                                                                                                             self.phone_featurizer.vocab_array)))
                    continue
                if self.check_valid(txt, self.text_featurizer.vocab_array) is not True:
                    logging.info(' {} txt  {} not all in tokens,continue'.format(txt, self.check_valid(txt,
                                                                                                             self.text_featurizer.vocab_array)))
                    continue

                txt = list(txt)
                phone_feature = self.phone_featurizer.extract(py)
                text_feature = self.text_featurizer.extract(txt)+[self.text_featurizer.endid()]

                if in_len < len(phone_feature):
                    logging.info('{} feature length < phone length,continue'.format(wp))
                    continue

                max_input = max(max_input, len(speech_feature))

                speech_features.append(speech_feature)
                input_length.append(in_len)
                phones.append(np.array(phone_feature))
                txts.append(np.array(text_feature))
                phones_length.append(len(phone_feature))


        if self.streaming:
            reduce = self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) * \
                     self.speech_config['stride_ms']
            max_input = max_input // self.chunk * self.chunk + self.chunk
            max_in_len = max_input // self.chunk
            chunk_times = self.chunk // reduce
            if self.chunk % reduce != 0:
                chunk_times += 1
            max_in_len *= chunk_times
            input_length = np.clip(input_length, 0, max_in_len)
        speech_features = self.speech_featurizer.pad_signal(speech_features, max_input)


        phones=tf.keras.preprocessing.sequence.pad_sequences(phones,maxlen=max([len(i) for i in phones]),padding='post',value=self.phone_featurizer.pad)
        txts=tf.keras.preprocessing.sequence.pad_sequences(txts,maxlen=max([len(i) for i in txts]),padding='post',value=self.text_featurizer.pad)
        x = np.array(speech_features, 'float32')
        phones = np.array(phones, 'int32')
        txts = np.array(txts, 'int32')

        input_length = np.array(input_length, 'int32')
        phones_length = np.array(phones_length, 'int32')

        return x, input_length, phones, phones_length,txts

    def generator(self, train=True):
        while 1:
            s=time.time()
            x, input_length, phones, phones_length,txts = self.generate(train)
            e=time.time()
            logging.info('load data cost time: {}'.format(e-s))
            if x.shape[0] == 0:
                logging.info('load data length zero,continue')
                continue


            yield x, input_length, phones, phones_length,txts
