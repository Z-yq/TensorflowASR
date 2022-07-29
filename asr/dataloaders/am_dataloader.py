import logging
import random

import numpy as np
import pypinyin
import jieba
import tensorflow as tf
import Pinyin2Hanzi
import jieba
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

        self.dagpms=Pinyin2Hanzi.DefaultDagParams()
    def return_data_types(self):

        return (tf.float32, tf.int32, tf.int32, tf.int32,
                tf.int32,
                tf.int32,
                tf.int32,
                tf.int32,
                tf.int32,
                tf.int32,
                tf.int32,
                )

    def return_data_shape(self):

        return (
            tf.TensorShape([self.batch, None, 1]),

            tf.TensorShape([self.batch, ]),
            tf.TensorShape([self.batch, None]),
            tf.TensorShape([self.batch, ]),
            tf.TensorShape([self.batch, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([self.batch, None]),
            tf.TensorShape([self.batch, None]),
            tf.TensorShape([self.batch, None]),
            tf.TensorShape([self.batch, None]),
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
        extra_text_list=self.speech_config['extra_text_list']
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
           
            if extra_text_list!='':
                with open(extra_text_list, encoding='utf-8') as f:
                    train_list = f.readlines()
                train_list = [i.strip() for i in train_list if i != '']
                np.random.shuffle(self.train_list)
                self.text_extra_list=train_list
                self.extra_text_offset=0
            else:
                self.text_extra_list=[]
                self.extra_text_offset = 0
            logging.info('load wav train list {}, test list {}, extra text list {}'.format(len(self.train_list), len(self.test_list),len(self.text_extra_list)))
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
    def hot_words_collection(self,txt,hot_words):

  

        cuts = jieba.lcut(txt)
        for cut in cuts:
            if len(cut) >= 2:
                new_word = Pinyin2Hanzi.dag(self.dagpms, pypinyin.lazy_pinyin(cut))
                for word in new_word:
                    word = ''.join(word.path)
                    if  cut != word:
                        if self.check_valid(word,self.text_featurizer.token_to_index) is True:
                            txt=txt.replace(cut,word)
                            hot_words.append(word)
                            break
        return txt,hot_words


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
        hot_words = []
        hot_words_phones = []
        hot_words_txts = []
        extra_text_phones=[]
        extra_text_txts=[]
        extra_text_hot_words_txts=[]
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
            if self.check_valid(py, self.phone_featurizer.token_to_index) is not True:
                logging.info(' {} txt phone {} not all in tokens,continue'.format(txt, self.check_valid(py,
                                                                                                        self.phone_featurizer.token_to_index)))
                continue
            if self.check_valid(txt, self.text_featurizer.token_to_index) is not True:
                logging.info(' {} txt  {} not all in tokens,continue'.format(txt, self.check_valid(txt,
                                                                                                        self.text_featurizer.token_to_index)))
                continue

            hot_words_txt, hot_words = self.hot_words_collection(txt, hot_words)
            txt = list(txt)
            phone_feature = self.phone_featurizer.extract(py)
            text_feature = self.text_featurizer.extract(txt) + [self.text_featurizer.endid()]
            hot_words_txt_feature = self.text_featurizer.extract(hot_words_txt) + [self.text_featurizer.endid()]

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
            hot_words_phones.append(np.array(phone_feature))
            hot_words_txts.append(np.array(hot_words_txt_feature))
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

                hot_words_txt, hot_words = self.hot_words_collection(txt, hot_words)
                txt = list(txt)
                phone_feature = self.phone_featurizer.extract(py)
                text_feature = self.text_featurizer.extract(txt) + [self.text_featurizer.endid()]
                hot_words_txt_feature = self.text_featurizer.extract(hot_words_txt) + [self.text_featurizer.endid()]

                if in_len < len(phone_feature):
                    logging.info('{} feature length < phone length,continue'.format(wp))
                    continue

                max_input = max(max_input, len(speech_feature))

                speech_features.append(speech_feature)
                input_length.append(in_len)
                phones.append(np.array(phone_feature))
                txts.append(np.array(text_feature))
                phones_length.append(len(phone_feature))
                hot_words_phones.append(np.array(phone_feature))
                hot_words_txts.append(np.array(hot_words_txt_feature))
        if len(self.text_extra_list)>0:
            for i in range(self.batch*10):
                txt=self.text_extra_list[self.extra_text_offset]
                self.extra_text_offset+=1
                if self.extra_text_offset>=len(self.text_extra_list):
                    self.extra_text_offset=0
                    np.random.shuffle(self.text_extra_list)
                if self.speech_config['only_chinese']:
                    txt = self.only_chinese(txt)

                py = self.text_to_vocab(txt)
                if self.check_valid(py, self.phone_featurizer.vocab_array) is not True:
                    logging.info(' {} txt phone {} not all in tokens,continue'.format(txt, self.check_valid(py,
                                                                                                            self.phone_featurizer.vocab_array)))
                    continue
                if self.check_valid(txt, self.text_featurizer.vocab_array) is not True:
                    logging.info(' {} txt  {} not all in tokens,continue'.format(txt, self.check_valid(txt,
                                                                                                       self.text_featurizer.vocab_array)))
                    continue

                hot_words_txt, hot_words = self.hot_words_collection(txt, hot_words)
                txt = list(txt)
                phone_feature = self.phone_featurizer.extract(py)
                text_feature = self.text_featurizer.extract(txt) + [self.text_featurizer.endid()]
                hot_words_txt_feature = self.text_featurizer.extract(hot_words_txt) + [self.text_featurizer.endid()]

                extra_text_hot_words_txts.append(np.array(hot_words_txt_feature))
                extra_text_phones.append(np.array(phone_feature))
                extra_text_txts.append(np.array(text_feature))
                if len(extra_text_txts)==self.batch:
                    break
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
        hot_words = list(set(hot_words))
        hot_words = [np.array(self.text_featurizer.extract(i)) for i in hot_words]

        phones = tf.keras.preprocessing.sequence.pad_sequences(phones, maxlen=max([len(i) for i in phones]),
                                                               padding='post', value=self.phone_featurizer.pad)
        txts = tf.keras.preprocessing.sequence.pad_sequences(txts, maxlen=max([len(i) for i in txts]), padding='post',
                                                             value=self.text_featurizer.pad)

        if len(extra_text_phones)>0:
            extra_text_phones = tf.keras.preprocessing.sequence.pad_sequences(extra_text_phones, maxlen=max([len(i) for i in extra_text_phones]),
                                                                   padding='post', value=self.phone_featurizer.pad)
            extra_text_txts = tf.keras.preprocessing.sequence.pad_sequences(extra_text_txts, maxlen=max([len(i) for i in extra_text_txts]), padding='post',
                                                                 value=self.text_featurizer.pad)
            extra_text_hot_words_txts = tf.keras.preprocessing.sequence.pad_sequences(extra_text_hot_words_txts, maxlen=max([len(i) for i in extra_text_hot_words_txts]), padding='post',
                                                                 value=self.text_featurizer.pad)
        
        hot_words = tf.keras.preprocessing.sequence.pad_sequences(hot_words, maxlen=max([len(i) for i in hot_words]),
                                                                  padding='post',
                                                                  value=self.text_featurizer.pad)

        hot_words_phones = tf.keras.preprocessing.sequence.pad_sequences(hot_words_phones,
                                                                       maxlen=max([len(i) for i in hot_words_phones]),
                                                                       padding='post',
                                                                       value=self.text_featurizer.pad)
        hot_words_txts = tf.keras.preprocessing.sequence.pad_sequences(hot_words_txts,
                                                                       maxlen=max([len(i) for i in hot_words_txts]),
                                                                       padding='post',
                                                                       value=self.text_featurizer.pad)
        x = np.array(speech_features, 'float32')
        phones = np.array(phones, 'int32')
        txts = np.array(txts, 'int32')
        hot_words = np.array(hot_words, 'int32')
        # hot_words=np.array([hot_words]*self.batch,'int32')
        hot_words_phones = np.array(hot_words_phones, 'int32')
        hot_words_txts = np.array(hot_words_txts, 'int32')

        input_length = np.array(input_length, 'int32')
        phones_length = np.array(phones_length, 'int32')
        if len(extra_text_txts)>0:
            extra_text_phones = np.array(extra_text_phones, 'int32')
            extra_text_txts = np.array(extra_text_txts, 'int32')
            extra_text_hot_words_txts = np.array(extra_text_hot_words_txts, 'int32')
        else:
            extra_text_phones=phones
            extra_text_txts=txts
            extra_text_hot_words_txts=txts
        return x, input_length, phones, phones_length, txts, hot_words,hot_words_phones, hot_words_txts,extra_text_phones,extra_text_txts,extra_text_hot_words_txts



    def generator(self, train=True):
        while 1:
            s=time.time()
            x, input_length, phones, phones_length, txts, hot_words,hot_words_phones, hot_words_txts,extra_text_phones,extra_text_txts,extra_text_hot_words_txts= self.generate(train)
            e=time.time()
            logging.info('load data cost time: {}'.format(e-s))
            if x.shape[0] == 0:
                logging.info('load data length zero,continue')
                continue

            # print(hot_words.shape,hot_words_phones.shape,hot_words_txts.shape,extra_text_txts.shape,extra_text_phones.shape)
            yield x, input_length, phones, phones_length, txts, hot_words,hot_words_phones, hot_words_txts,extra_text_phones,extra_text_txts,extra_text_hot_words_txts

    def eval_data_generator(self):

        sample = []
        speech_features = []
        input_length = []
        phones = []
        phones_length = []
        txts = []
        max_input = 0
        hot_words = []
        hot_words_txts = []
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
                logging.info(' {} txt phone {} not all in tokens,skip'.format(txt, self.check_valid(py,
                                                                                                    self.phone_featurizer.vocab_array)))
                continue
            if self.check_valid(txt, self.text_featurizer.vocab_array) is not True:
                logging.info(' {} txt phone {} not all in tokens,skip'.format(txt, self.check_valid(py,
                                                                                                    self.text_featurizer.vocab_array)))
                continue
            hot_words_txt, hot_words = self.hot_words_collection(txt, hot_words)
            txt = list(txt)
            phone_feature = self.phone_featurizer.extract(py)
            text_feature = self.text_featurizer.extract(txt) + [self.text_featurizer.endid()]
            hot_words_txt_feature = self.text_featurizer.extract(hot_words_txt) + [self.text_featurizer.endid()]

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
            hot_words_txts.append(np.array(hot_words_txt_feature))
            if len(sample) == batch:
                break

        hot_words = list(set(hot_words))
        hot_words = [np.array(self.text_featurizer.extract(i)) for i in hot_words]
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

        hot_words = tf.keras.preprocessing.sequence.pad_sequences(hot_words, maxlen=max([len(i) for i in hot_words]),
                                                                  padding='post',
                                                                  value=self.text_featurizer.pad)

        hot_words_txts = tf.keras.preprocessing.sequence.pad_sequences(hot_words_txts,
                                                                       maxlen=max([len(i) for i in hot_words_txts]),
                                                                       padding='post',
                                                                       value=self.text_featurizer.pad)
        x = np.array(speech_features, 'float32')
        phones = np.array(phones, 'int32')
        txts = np.array(txts, 'int32')
        hot_words = np.array(hot_words, 'int32')
        hot_words_txts = np.array(hot_words_txts, 'int32')

        input_length = np.array(input_length, 'int32')
        phones_length = np.array(phones_length, 'int32')

        return x, input_length, phones, phones_length, txts, hot_words, hot_words_txts