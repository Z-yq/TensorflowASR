from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer
import numpy as np
from augmentations.augments import Augmentation
import random
import os
from jieba.posseg import lcut
from keras_bert import Tokenizer, load_vocabulary, load_trained_model_from_checkpoint
import tensorflow as tf
import pypinyin
class E2E_DataLoader():

    def __init__(self, config_dict, training=True):
        self.speech_config = config_dict['speech_config']
        self.text_config = config_dict['text_config']
        self.text_py_config = config_dict['pinyin_decoder_config']
        self.text_ch_config = config_dict['zh_decoder_config']
        self.augment_config = config_dict['augments_config']
        self.batch = config_dict['learning_config']['running_config']['batch_size']
        self.speech_featurizer = SpeechFeaturizer(self.speech_config)
        self.token_py_featurizer = TextFeaturizer(self.text_py_config)
        self.token_ch_featurizer = TextFeaturizer(self.text_ch_config)

        self.make_file_list(self.speech_config['train_list'] if training else self.speech_config['eval_list'],
                            self.text_config['extra_train_list'] if training else self.text_config['extra_eval_list'],
                            training)
        self.make_maps(config_dict)
        self.augment = Augmentation(self.augment_config)
        self.epochs = 1
        self.steps = 0

        self.init_bert(config_dict)

    def load_state(self, outdir):
        try:
            self.pick_index = np.load(os.path.join(outdir, 'dg_state.npy')).flatten().tolist()
            self.epochs = 1 + int(np.mean(self.wav_pick_index))
        except FileNotFoundError:
            print('not found state file')
        except:
            print('load state falied,use init state')

    def save_state(self, outdir):
        np.save(os.path.join(outdir, 'dg_state.npy'), np.array(self.wav_pick_index))

    def load_bert(self, config, checkpoint):
        model = load_trained_model_from_checkpoint(config, checkpoint, trainable=False, seq_len=None)
        return model

    def init_bert(self, config):
        bert_config = config['bert_config']['config_json']
        bert_checkpoint = config['bert_config']['bert_ckpt']
        bert_vocab = config['bert_config']['bert_vocab']
        bert_vocabs = load_vocabulary(bert_vocab)
        self.bert_token = Tokenizer(bert_vocabs)
        self.bert = self.load_bert(bert_config, bert_checkpoint)

    def bert_decode(self, x):
        tokens, segs = [], []

        for i in x:
            t, s = self.bert_token.encode(''.join(i))
            tokens.append(t)
            segs.append(s)
        return tokens, segs

    def get_bert_feature(self, bert_t, bert_s):
        f = []
        for t, s in zip(bert_t, bert_s):
            t = np.expand_dims(np.array(t), 0)
            s = np.expand_dims(np.array(s), 0)
            feature = self.bert.predict([t, s])
            f.append(feature[0])
        return f[0][1:]

    def return_data_types(self):

        return (
        tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
        tf.int32, tf.int32, tf.float32)

    def return_data_shape(self):
        f, c = self.speech_featurizer.compute_feature_dim()

        return (
            tf.TensorShape([None, None, f, c]),
            tf.TensorShape([None, None, 1]),
            tf.TensorShape([None, None, 768]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None, None, None])
        )

    def get_per_epoch_steps(self):
        return len(self.wav_train_list) // self.batch

    def eval_per_epoch_steps(self):
        return len(self.wav_test_list) // self.batch

    def make_maps(self, config):
        with open(config['map_path']['pinyin'], encoding='utf-8') as f:
            data = f.readlines()
        data = [i.strip() for i in data if i != '']
        self.py_map = {}
        for line in data:
            key, py = line.strip().split('\t')
            self.py_map[key] = py
            if len(py.split(' ')) > 1:
                for i, j in zip(list(key), py.split(' ')):
                    self.py_map[i] = j
        with open(config['map_path']['phone'], encoding='utf-8') as f:
            data = f.readlines()
        data = [i.strip() for i in data if i != '']
        self.phone_map = {}
        phone_map = {}
        for line in data:
            key, py = line.strip().split('\t')
            phone_map[key] = py
        for key in self.py_map.keys():
            key_py = self.py_map[key]
            if len(key) > 1:
                phone = []
                for n in key_py.split(' '):
                    phone += [phone_map[n]]
                self.phone_map[key] = ' '.join(phone)
            else:
                self.phone_map[key] = phone_map[self.py_map[key]]

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

    def make_file_list(self, wav_list, text_list,training=True):
        with open(wav_list, encoding='utf-8') as f:
            data = f.readlines()
        data = [i.strip() for i in data if i != '']
        num = len(data)
        if training:
            self.wav_train_list = data[:int(num * 0.99)]
            self.wav_test_list = data[int(num * 0.99):]
            np.random.shuffle(self.wav_train_list)
            self.wav_pick_index = [0.] * len(self.wav_train_list)
            if text_list!='':
                with open(text_list, encoding='utf-8') as f:
                    datas = f.readlines()
                datas = [i.strip() for i in datas if i != '']
            else:
                datas=[i.strip().split('\t')[1] for i in data]
            num = len(datas)
            self.lm_train_list=datas[:int(num * 0.99)]
            self.lm_test_list = datas[int(num * 0.99):]
        else:
            self.wav_eval_list = data
            if text_list!='':
                with open(text_list, encoding='utf-8') as f:
                    datas = f.readlines()
                datas = [i.strip() for i in datas if i != '']
            else:
                datas=[i.strip().split('\t')[1] for i in data]
            self.lm_eval_list=datas
            self.wav_offset = 0

    def only_chinese(self, word):
        new=''
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                new+=ch
        return new

    def eval_data_generator(self):
        sample = self.wav_test_list[self.wav_offset:self.wav_offset + self.batch]
        self.wav_offset += self.batch
        speech_features = []
        input_length = []
        py_label = []
        py_label_length = []
        txt_label = []
        txt_label_length = []
        bert_features = []
        max_wav = 0
        max_input = 0
        max_label_py = 0
        max_label_txt = 0
        for i in sample:
            wp, txt = i.strip().split('\t')
            try:
                data = self.speech_featurizer.load_wav(wp)
            except:
                print('load data failed')
                continue
            if len(data) < 400:
                continue
            elif len(data) > self.speech_featurizer.sample_rate * 7:
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
            py = self.text_to_vocab(txt)
            if len(py) == 0:
                continue
            e_bert_t, e_bert_s = self.bert_decode([txt])
            bert_feature = self.get_bert_feature(e_bert_t, e_bert_s)

            py_text_feature = self.token_py_featurizer.extract(py)
            ch_text_feature = self.token_ch_featurizer.extract(list(txt))

            if speech_feature.shape[0] // self.speech_config['reduction_factor'] < len(py_text_feature) or \
                    speech_feature.shape[0] // self.speech_config['reduction_factor'] < len(ch_text_feature):
                continue
            max_input = max(max_input, speech_feature.shape[0])

            max_label_py = max(max_label_py, len(py_text_feature))
            max_label_txt = max(max_label_txt, len(ch_text_feature))

            max_wav = max(max_wav, len(data))
            speech_features.append(speech_feature)

            input_length.append(in_len)

            py_label.append(np.array(py_text_feature))
            py_label_length.append(len(py_text_feature))

            txt_label.append(np.array(ch_text_feature))
            txt_label_length.append(len(ch_text_feature))
            bert_features.append(bert_feature)

        if self.speech_config['use_mel_layer']:
            speech_features = self.speech_featurizer.pad_signal(speech_features, max_wav)

        else:
            for i in range(len(speech_features)):

                if speech_features[i].shape[0] < max_input:
                    pad = np.ones([max_input - speech_features[i].shape[0], speech_features[i].shape[1],
                                   speech_features[i].shape[2]]) * speech_features[i].min()
                    speech_features[i] = np.vstack((speech_features[i], pad))
        for i in range(len(bert_features)):
            if bert_features[i].shape[0] < max_label_txt:
                pading = np.ones([max_label_txt - len(bert_features[i]), 768]) * -10.
                bert_features[i] = np.vstack((bert_features[i], pading))

        py_label = self.pad(py_label, max_label_py)
        txt_label = self.pad(txt_label, max_label_txt)

        speech_features = np.array(speech_features, 'float32')
        bert_features = np.array(bert_features, 'float32')

        py_label = np.array(py_label, 'int32')
        txt_label = np.array(txt_label, 'int32')

        input_length = np.array(input_length, 'int32')
        py_label_length = np.array(py_label_length, 'int32')
        txt_label_length = np.array(txt_label_length, 'int32')

        return speech_features, bert_features, input_length, py_label, py_label_length, txt_label, txt_label_length
    def pad(self, words_label, max_label_words):
        for i in range(len(words_label)):
            if words_label[i].shape[0] < max_label_words:
                pad = np.ones(max_label_words - words_label[i].shape[0]) * self.token_py_featurizer.pad
                words_label[i] = np.hstack((words_label[i], pad))
        return words_label

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

    def generate_speech_data(self, train=True):

        if train:
            batch = self.batch if self.augment.available() else self.batch * 2
            indexs = np.argsort(self.wav_pick_index)[:batch]
            indexs = random.sample(indexs.tolist(), batch // 2)
            sample = [self.wav_train_list[i] for i in indexs]
            for i in indexs:
                self.wav_pick_index[int(i)] += 1
            self.epochs = 1 + int(np.mean(self.wav_pick_index))
        else:
            sample = random.sample(self.wav_test_list, self.batch)

        speech_features = []
        input_length = []
        py_label = []
        py_label_length = []
        txt_label = []
        txt_label_length = []
        bert_features = []
        max_wav = 0
        max_input = 0
        max_label_py = 0
        max_label_txt = 0
        for i in sample:
            wp, txt = i.strip().split('\t')
            try:
                data = self.speech_featurizer.load_wav(wp)
            except:
                print('load data failed')
                continue
            if len(data) < 400:
                continue
            elif len(data) > self.speech_featurizer.sample_rate * 7:
                continue

            if self.speech_config['only_chinese']:
                txt = self.only_chinese(txt)
            if self.speech_config['use_mel_layer']:
                speech_feature = data / np.abs(data).max()
                speech_feature = np.expand_dims(speech_feature, -1)
                in_len=len(speech_feature) // (self.speech_config['reduction_factor']*(self.speech_featurizer.sample_rate/1000)*self.speech_config['stride_ms'])
            else:
                speech_feature = self.speech_featurizer.extract(data)
                in_len=int(speech_feature.shape[0]//self.speech_config['reduction_factor'])
            py = self.text_to_vocab(txt)
            if len(py) == 0:
                continue
            e_bert_t, e_bert_s = self.bert_decode([txt])
            bert_feature = self.get_bert_feature(e_bert_t, e_bert_s)

            py_text_feature = self.token_py_featurizer.extract(py)
            ch_text_feature = self.token_ch_featurizer.extract(list(txt))

            if speech_feature.shape[0] // self.speech_config['reduction_factor'] < len(py_text_feature) or \
                    speech_feature.shape[0] // self.speech_config['reduction_factor'] < len(ch_text_feature) :
                continue
            max_input = max(max_input, speech_feature.shape[0])

            max_label_py = max(max_label_py, len(py_text_feature))
            max_label_txt = max(max_label_txt, len(ch_text_feature))

            max_wav = max(max_wav, len(data))
            speech_features.append(speech_feature)

            input_length.append(in_len)


            py_label.append(np.array(py_text_feature))
            py_label_length.append(len(py_text_feature))

            txt_label.append(np.array(ch_text_feature))
            txt_label_length.append(len(ch_text_feature))
            bert_features.append(bert_feature)

        if train and self.augment.available():
            for i in sample:

                wp, txt = i.strip().split('\t')
                try:
                    data = self.speech_featurizer.load_wav(wp)
                except:
                    print('load data failed')
                    continue
                if len(data) < 400:
                    continue
                elif len(data) > self.speech_featurizer.sample_rate * 7:
                    continue
                
                if self.speech_config['only_chinese']:
                    txt = self.only_chinese(txt)

                if self.speech_config['use_mel_layer']:
                    speech_feature = data / np.abs(data).max()
                    speech_feature = np.expand_dims(speech_feature,-1)
                    in_len = len(speech_feature) // (
                                self.speech_config['reduction_factor'] * (self.speech_featurizer.sample_rate / 1000) *
                                self.speech_config['stride_ms'])
                else:
                    speech_feature = self.speech_featurizer.extract(data)
                    in_len = int(speech_feature.shape[0] // self.speech_config['reduction_factor'])

                py = self.text_to_vocab(txt)
                if len(py) == 0:
                    continue
                e_bert_t, e_bert_s = self.bert_decode([txt])
                bert_feature = self.get_bert_feature(e_bert_t, e_bert_s)

                py_text_feature = self.token_py_featurizer.extract(py)
                ch_text_feature = self.token_ch_featurizer.extract(list(txt))

                if speech_feature.shape[0] // self.speech_config['reduction_factor'] < len(py_text_feature) or \
                        speech_feature.shape[0] // self.speech_config['reduction_factor'] < len(ch_text_feature):
                    continue
                max_input = max(max_input, speech_feature.shape[0])

                max_label_py = max(max_label_py, len(py_text_feature))
                max_label_txt = max(max_label_txt, len(ch_text_feature))

                max_wav = max(max_wav, len(data))
                speech_features.append(speech_feature)

                input_length.append(in_len)

                py_label.append(np.array(py_text_feature))
                py_label_length.append(len(py_text_feature))

                txt_label.append(np.array(ch_text_feature))
                txt_label_length.append(len(ch_text_feature))
                bert_features.append(bert_feature)
        if self.speech_config['use_mel_layer']:
            speech_features = self.speech_featurizer.pad_signal(speech_features, max_wav)

        else:
            for i in range(len(speech_features)):

                if speech_features[i].shape[0] < max_input:
                    pad = np.ones([max_input - speech_features[i].shape[0], speech_features[i].shape[1], speech_features[i].shape[2]]) * speech_features[i].min()
                    speech_features[i] = np.vstack((speech_features[i], pad))
        for i in range(len(bert_features)):
            if bert_features[i].shape[0] < max_label_txt:
                pading = np.ones([max_label_txt - len(bert_features[i]), 768]) * -10.
                bert_features[i] = np.vstack((bert_features[i], pading))


        py_label = self.pad(py_label, max_label_py)
        txt_label = self.pad(txt_label, max_label_txt)

        speech_features = np.array(speech_features, 'float32')
        bert_features = np.array(bert_features, 'float32')

        py_label = np.array(py_label, 'int32')
        txt_label = np.array(txt_label, 'int32')

        input_length = np.array(input_length, 'int32')
        py_label_length = np.array(py_label_length, 'int32')
        txt_label_length = np.array(txt_label_length, 'int32')


        return speech_features, bert_features, input_length,  py_label, py_label_length, txt_label, txt_label_length

    def preprocess(self, tokens, txts):
        x = []
        y = []
        for token, txt in zip(tokens, txts):
            # print(py,txt)
            # try:
            x_ = [self.token_py_featurizer.startid()]
            y_ = [self.token_ch_featurizer.startid()]
            for i in token:
                x_.append(self.token_py_featurizer.token_to_index[i])
            for i in txt:
                y_.append(self.token_ch_featurizer.token_to_index[i])
            x_.append(self.token_py_featurizer.endid())
            y_.append(self.token_ch_featurizer.endid())
            x.append(np.array(x_))
            y.append(np.array(y_))

        return x, y

    def generate_text_data(self,train=True):
        if train:
            sample = random.sample(self.lm_train_list, self.batch)
        else:
            sample = random.sample(self.lm_test_list, self.batch)
        trainx = [self.text_to_vocab(i) for i in sample]
        trainy = sample
        x, y = self.preprocess(trainx, trainy)
        e_bert_t, e_bert_s = self.bert_decode(trainy)
        bert_features = self.get_bert_feature(e_bert_t, e_bert_s)
        x_max=max([len(i) for i in x])
        y_max=max([len(i) for i in y])
        x = self.pad(x,x_max)
        y = self.pad(y,y_max)
        max_label_len=max([len(i) for i in bert_features])
        for i in range(len(bert_features)):
            if bert_features[i].shape[0] < max_label_len:
                pading = np.ones([max_label_len - len(bert_features[i]), 768]) * -10.
                bert_features[i] = np.vstack((bert_features[i], pading))

        x = np.array(x)
        y = np.array(y)
        bert_features = np.array(bert_features, dtype='float32')

        return x, y, bert_features

    def generator(self, train=True):
        while 1:
            speech_features, bert_features, input_length,  py_label, py_label_length, txt_label, txt_label_length = self.generate_speech_data(
                train)

            guide_matrix = self.guided_attention(input_length, txt_label_length, np.max(input_length),
                                                 txt_label_length.max())
            yield speech_features, bert_features, input_length,  py_label, py_label_length, txt_label, txt_label_length, guide_matrix


