from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer
import numpy as np
from augmentations.augments import Augmentation
import random
import os
import pypinyin
import tensorflow as tf


class MultiTask_DataLoader():

    def __init__(self, config_dict, training=True):
        self.speech_config = config_dict['speech_config']
        self.text1_config = config_dict['decoder1_config']
        self.text2_config = config_dict['decoder2_config']
        self.text3_config = config_dict['decoder3_config']
        self.augment_config = config_dict['augments_config']
        self.batch = config_dict['learning_config']['running_config']['batch_size']
        self.speech_featurizer = SpeechFeaturizer(self.speech_config)
        self.token1_featurizer = TextFeaturizer(self.text1_config)
        self.token2_featurizer = TextFeaturizer(self.text2_config)
        self.token3_featurizer = TextFeaturizer(self.text3_config)
        self.make_file_list(self.speech_config['train_list'] if training else self.speech_config['eval_list'], training)
        self.make_maps(config_dict)
        self.augment = Augmentation(self.augment_config)
        self.epochs = 1
        self.steps = 0

    def load_state(self, outdir):
        try:

            dg_state = np.load(os.path.join(outdir, 'dg_state.npz'))

            self.epochs = int(dg_state['epoch'])
            self.train_offset = int(dg_state['train_offset'])
            train_list = dg_state['train_list'].tolist()
            if len(train_list)!=len(self.train_list):
                print('history train list not equal train list ,data loader use init state')
                self.epochs=0
                self.train_offset=0
        except FileNotFoundError:
            print('not found state file,init state')
        except:
            print('load state falied,use init state')

    def save_state(self, outdir):
        # np.save(os.path.join(outdir, 'dg_state.npy'), np.array(self.pick_index))
        np.savez(os.path.join(outdir,'dg_state.npz'),epoch=self.epochs,train_offset=self.train_offset,train_list=self.train_list)

    def return_data_types(self):

        return (
        tf.float32,  tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)

    def return_data_shape(self):
        f, c = self.speech_featurizer.compute_feature_dim()

        return (
            tf.TensorShape([None, None, 1]) if self.speech_config['use_mel_layer'] else tf.TensorShape(
                [None, None, f, c]),

            tf.TensorShape([None, ]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, ]),
        )

    def get_per_epoch_steps(self):
        return len(self.train_list) // self.batch

    def eval_per_epoch_steps(self):
        return len(self.test_list) // self.batch

    def make_maps(self, config):
        with open(config['map_path']['phone'], encoding='utf-8') as f:
            data = f.readlines()
        data = [i.strip() for i in data if i != '']
        self.phone_map = {}
        phone_map = {}
        for line in data:
            try:
                key, phone = line.strip().split('\t')
            except:
                continue
            phone_map[key] = phone.split(' ')
        self.phone_map=phone_map

    def map(self, txt):
        pys=pypinyin.pinyin(txt,8,neutral_tone_with_five=True)

        pys = [i[0] for i in pys]
        phones = []
       

        for i in pys:
            phones+=self.phone_map[i]
        words=''.join(pys)
        words=list(words)
        return pys, phones, words

    def augment_data(self, wavs, label, label_length):
        if not self.augment.available():
            return None
        mels = []
        input_length = []
        label_ = []
        label_length_ = []
        wavs_ = []
        max_input = 0
        max_wav = 0
        for idx, wav in enumerate(wavs):

            data = self.augment.process(wav.flatten())
            speech_feature = self.speech_featurizer.extract(data)
            if speech_feature.shape[0] // self.speech_config['reduction_factor'] < label_length[idx]:
                continue
            max_input = max(max_input, speech_feature.shape[0])

            max_wav = max(max_wav, len(data))

            wavs_.append(data)

            mels.append(speech_feature)
            input_length.append(speech_feature.shape[0] // self.speech_config['reduction_factor'])
            label_.append(label[idx])
            label_length_.append(label_length[idx])

        for i in range(len(mels)):
            if mels[i].shape[0] < max_input:
                pad = np.ones([max_input - mels[i].shape[0], mels[i].shape[1], mels[i].shape[2]]) * mels[i].min()
                mels[i] = np.vstack((mels[i], pad))

        wavs_ = self.speech_featurizer.pad_signal(wavs_, max_wav)

        x = np.array(mels, 'float32')
        label_ = np.array(label_, 'int32')

        input_length = np.array(input_length, 'int32')
        label_length_ = np.array(label_length_, 'int32')

        wavs_ = np.array(np.expand_dims(wavs_, -1), 'float32')

        return x, wavs_, input_length, label_, label_length_

    def make_file_list(self, wav_list, training=True):
        with open(wav_list, encoding='utf-8') as f:
            data = f.readlines()
        data = [i.strip() for i in data if i != '']
        num = len(data)
        if training:
            self.train_list = data[:int(num * 0.99)]
            self.test_list = data[int(num * 0.99):]
            np.random.shuffle(self.train_list)
            self.train_offset=0
            self.test_offset=0
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
    def check_valid(self,txt,vocab_list):
        if len(txt)==0:
            return False
        for n in txt:
            if n in vocab_list:
                pass
            else:
                return n
        return True
    def eval_data_generator(self):
        sample = self.test_list[self.offset:self.offset + self.batch]
        self.offset += self.batch
        speech_features = []
        input_length = []

        words_label = []
        words_label_length = []

        phone_label = []
        phone_label_length = []

        py_label = []
        py_label_length = []

        max_input = 0
        max_label_words = 0
        max_label_phone = 0
        max_label_py = 0

        for i in sample:
            wp, txt = i.strip().split('\t')
            try:
                data = self.speech_featurizer.load_wav(wp)
            except:
                print('{} load data failed,skip'.format(wp))
                continue
            if len(data) < 400:
                continue
            elif len(data) > self.speech_featurizer.sample_rate * self.speech_config['wav_max_duration']:
                print('{} duration out of wav_max_duration({}),skip'.format(wp, self.speech_config['wav_max_duration']))
                continue
            if self.speech_config['only_chinese']:
                txt = self.only_chinese(txt)
            if self.speech_config['use_mel_layer']:
                speech_feature = data / np.abs(data).max()
                speech_feature = np.expand_dims(speech_feature, -1)
                in_len = len(speech_feature) // (
                        self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) *
                        self.speech_config['stride_ms'])
            else:
                speech_feature = self.speech_featurizer.extract(data)
                in_len = int(speech_feature.shape[0] // self.speech_config['reduction_factor'])

            py, phone, word = self.map(txt)
            if len(py) == 0:
                continue

            if not self.check_valid(word, self.token1_featurizer.vocab_array):
                print(' {} txt word {} not all in tokens,continue'.format(txt, py))
                continue
                
            if not self.check_valid(phone, self.token1_featurizer.vocab_array):
                print(' {} txt phone {} not all in tokens,continue'.format(txt, py))
                continue
                
            if not self.check_valid(py, self.token1_featurizer.vocab_array):
                print(' {} txt pinyin {} not all in tokens,continue'.format(txt, py))
                continue
            word_text_feature = self.token1_featurizer.extract(word)
            phone_text_feature = self.token2_featurizer.extract(phone)
            py_text_feature = self.token3_featurizer.extract(py)
          
            if in_len  < len(word_text_feature):
                continue
            
            max_label_words = max(max_label_words, len(word_text_feature))
            max_label_phone = max(max_label_phone, len(phone_text_feature))
            max_label_py = max(max_label_py, len(py_text_feature))
            max_input = max(max_input, len(speech_feature))

            speech_features.append(speech_feature)
            input_length.append(in_len)
            words_label.append(np.array(word_text_feature))
            words_label_length.append(len(word_text_feature))

            phone_label.append(np.array(phone_text_feature))
            phone_label_length.append(len(phone_text_feature))

            py_label.append(np.array(py_text_feature))
            py_label_length.append(len(py_text_feature))

           

        
        if self.speech_config['use_mel_layer']:
            speech_features = self.speech_featurizer.pad_signal(speech_features, max_input)

        else:
            for i in range(len(speech_features)):

                if speech_features[i].shape[0] < max_input:
                    pad = np.ones([max_input - speech_features[i].shape[0], speech_features[i].shape[1],
                                   speech_features[i].shape[2]]) * speech_features[i].min()
                    speech_features[i] = np.vstack((speech_features[i], pad))




        words_label = self.pad(words_label, max_label_words)
        phone_label = self.pad(phone_label, max_label_phone)
        py_label = self.pad(py_label, max_label_py)
        speech_features = np.array(speech_features, 'float32')
        words_label = np.array(words_label, 'int32')
        phone_label = np.array(phone_label, 'int32')
        py_label = np.array(py_label, 'int32')
        input_length = np.array(input_length, 'int32')
        words_label_length = np.array(words_label_length, 'int32')
        phone_label_length = np.array(phone_label_length, 'int32')
        py_label_length = np.array(py_label_length, 'int32')

        return speech_features, input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length

    def pad(self, words_label, max_label_words):
        for i in range(len(words_label)):
            if words_label[i].shape[0] < max_label_words:
                pad = np.ones(max_label_words - words_label[i].shape[0]) * self.token1_featurizer.pad
                words_label[i] = np.hstack((words_label[i], pad))
        return words_label

    def GuidedAttention(self, N, T, g=0.2):
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
            att_target = self.GuidedAttention(i, step, 0.2)
            pad[:att_target.shape[0], :att_target.shape[1]] = att_target
            att_targets.append(pad)
        att_targets = np.array(att_targets)

        return att_targets.astype('float32')

    def generate(self, train=True):
        sample=[]
        speech_features = []
        input_length = []

        words_label = []
        words_label_length = []

        phone_label = []
        phone_label_length = []

        py_label = []
        py_label_length = []


        max_input = 0
        max_label_words = 0
        max_label_phone = 0
        max_label_py = 0
        if train:
            batch = self.batch//2 if self.augment.available() else self.batch
        else:
            batch=self.batch




        for i in range(batch*10):
            if train:
                line=self.train_list[self.train_offset]
                self.train_offset+=1
                if self.train_offset>len(self.train_list)-1:
                    self.train_offset=0
                    np.random.shuffle(self.train_list)
                    self.epochs+=1
            else:
                line = self.test_list[self.test_offset]
                self.test_offset += 1
                if self.test_offset > len(self.test_list) - 1:
                    self.test_offset = 0

            wp, txt = line.strip().split('\t')
            try:
                data = self.speech_featurizer.load_wav(wp)
            except:
                print('{} load data failed,skip'.format(wp))
                continue
            if len(data) < 400:
                continue
            elif len(data) > self.speech_featurizer.sample_rate * self.speech_config['wav_max_duration']:
                print('{} duration out of wav_max_duration({}),skip'.format(wp, self.speech_config['wav_max_duration']))
                continue
            if self.speech_config['only_chinese']:
                txt = self.only_chinese(txt)
            if self.speech_config['use_mel_layer']:
                speech_feature = data / np.abs(data).max()
                speech_feature = np.expand_dims(speech_feature, -1)
                in_len = len(speech_feature) // (
                        self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) *
                        self.speech_config['stride_ms'])
            else:
                speech_feature = self.speech_featurizer.extract(data)
                in_len = int(speech_feature.shape[0] // self.speech_config['reduction_factor'])

            py, phone, word = self.map(txt)
            if len(py) == 0:
                print('py length',len(py),'skip')
                continue

            if  self.check_valid(word, self.token1_featurizer.vocab_array) is not True:
                print(' {} txt word {} not all in tokens,continue'.format(txt,self.check_valid(word, self.token1_featurizer.vocab_array)))
                continue
            #
            if self.check_valid(phone, self.token2_featurizer.vocab_array) is not True:
                print(' {} txt phone {} not all in tokens,continue'.format(txt, self.check_valid(phone,
                                                                                                self.token2_featurizer.vocab_array)))
                continue
            #
            if self.check_valid(py, self.token3_featurizer.vocab_array) is not True:
                print(' {} txt py {} not all in tokens,continue'.format(txt, self.check_valid(py,
                                                                                                self.token3_featurizer.vocab_array)))
                continue
            word_text_feature = self.token1_featurizer.extract(word)
            phone_text_feature = self.token2_featurizer.extract(phone)
            py_text_feature = self.token3_featurizer.extract(py)

            if in_len < len(word_text_feature):
                continue

            max_label_words = max(max_label_words, len(word_text_feature))
            max_label_phone = max(max_label_phone, len(phone_text_feature))
            max_label_py = max(max_label_py, len(py_text_feature))
            max_input = max(max_input, len(speech_feature))

            speech_features.append(speech_feature)
            input_length.append(in_len)
            words_label.append(np.array(word_text_feature))
            words_label_length.append(len(word_text_feature))

            phone_label.append(np.array(phone_text_feature))
            phone_label_length.append(len(phone_text_feature))

            py_label.append(np.array(py_text_feature))
            py_label_length.append(len(py_text_feature))
            sample.append(line)
            if len(sample)==batch:
                break
        if train and self.augment.available():
            for i in sample:
                wp, txt = i.strip().split('\t')
                try:
                    data = self.speech_featurizer.load_wav(wp)
                except:
                    print('{} load data failed,skip'.format(wp))
                    continue
                if len(data) < 400:
                    continue
                elif len(data) > self.speech_featurizer.sample_rate * self.speech_config['wav_max_duration']:
                    print('{} duration out of wav_max_duration({}),skip'.format(wp,
                                                                                self.speech_config['wav_max_duration']))
                    continue
                data=self.augment.process(data)
                if self.speech_config['only_chinese']:
                    txt = self.only_chinese(txt)
                if self.speech_config['use_mel_layer']:
                    speech_feature = data / np.abs(data).max()
                    speech_feature = np.expand_dims(speech_feature, -1)
                    in_len = len(speech_feature) // (
                            self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) *
                            self.speech_config['stride_ms'])
                else:
                    speech_feature = self.speech_featurizer.extract(data)
                    in_len = int(speech_feature.shape[0] // self.speech_config['reduction_factor'])

                py, phone, word = self.map(txt)
                if len(py) == 0:
                    continue


                word_text_feature = self.token1_featurizer.extract(word)
                phone_text_feature = self.token2_featurizer.extract(phone)
                py_text_feature = self.token3_featurizer.extract(py)

                if in_len < len(word_text_feature):
                    continue

                max_label_words = max(max_label_words, len(word_text_feature))
                max_label_phone = max(max_label_phone, len(phone_text_feature))
                max_label_py = max(max_label_py, len(py_text_feature))
                max_input = max(max_input, len(speech_feature))

                speech_features.append(speech_feature)
                input_length.append(in_len)
                words_label.append(np.array(word_text_feature))
                words_label_length.append(len(word_text_feature))

                phone_label.append(np.array(phone_text_feature))
                phone_label_length.append(len(phone_text_feature))

                py_label.append(np.array(py_text_feature))
                py_label_length.append(len(py_text_feature))

        if self.speech_config['use_mel_layer']:
            speech_features = self.speech_featurizer.pad_signal(speech_features, max_input)

        else:
            for i in range(len(speech_features)):

                if speech_features[i].shape[0] < max_input:
                    pad = np.ones([max_input - speech_features[i].shape[0], speech_features[i].shape[1],
                                   speech_features[i].shape[2]]) * speech_features[i].min()
                    speech_features[i] = np.vstack((speech_features[i], pad))

        words_label = self.pad(words_label, max_label_words)
        phone_label = self.pad(phone_label, max_label_phone)
        py_label = self.pad(py_label, max_label_py)
        speech_features = np.array(speech_features, 'float32')
        words_label = np.array(words_label, 'int32')
        phone_label = np.array(phone_label, 'int32')
        py_label = np.array(py_label, 'int32')
        input_length = np.array(input_length, 'int32')
        words_label_length = np.array(words_label_length, 'int32')
        phone_label_length = np.array(phone_label_length, 'int32')
        py_label_length = np.array(py_label_length, 'int32')

        return speech_features, input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length

    def generator(self, train=True):
        while 1:
            speech_features, input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length= self.generate(train)

            yield speech_features, input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length