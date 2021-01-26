import os
import random
import logging
import numpy as np
import pypinyin
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras_bert import Tokenizer, load_vocabulary, load_trained_model_from_checkpoint

from utils.text_featurizers import TextFeaturizer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class LM_DataLoader():
    def __init__(self, config, training=True):
        self.train = training
        self.init_all(config)
        self.for_multi_task=config['am_token']['for_multi_task']
        self.am_featurizer = TextFeaturizer(config['am_token'])
        self.lm_featurizer = TextFeaturizer(config['lm_token'])
        self.init_text_to_vocab()
        self.batch = config['running_config']['batch_size']
        self.epochs = 1
        self.config=config

    def init_bert(self, config, checkpoint):
        model = load_trained_model_from_checkpoint(config, checkpoint, trainable=False, seq_len=None)
        return model

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

        return (tf.int32, tf.int32, tf.float32)

    def return_data_shape(self):

        return (
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None, 768])
        )

    def get_per_epoch_steps(self):
        return len(self.train_list) // self.batch

    def eval_per_epoch_steps(self):
        return len(self.test_texts) // self.batch

    def init_all(self, config):
        if self.train:
            bert_config = config['bert']['config_json']
            bert_checkpoint = config['bert']['bert_ckpt']
            bert_vocab = config['bert']['bert_vocab']
            bert_vocabs = load_vocabulary(bert_vocab)
            self.bert_token = Tokenizer(bert_vocabs)
            self.bert = self.init_bert(bert_config, bert_checkpoint)
        self.get_sentence(config['train_list'] if self.train else config['eval_list'], training=self.train)

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
            if self.for_multi_task:
                pys = pypinyin.pinyin(txt, 8, neutral_tone_with_five=True)
                pys = [i[0] for i in pys]
                return pys
            else:
                pys=pypinyin.pinyin(txt)
                pys=[i[0] for i in pys]
                return pys

        self.text_to_vocab = text_to_vocab_func

    def get_sentence(self, data_path, training):
        from tqdm import tqdm

        with open(data_path, encoding='utf-8') as f:
            data = f.readlines()

        txts = []
        for txt in tqdm(data):
            txt = txt.strip()
            if len(txt) > 150:
                continue
            txts.append(txt)
        if training:
            num = len(txts)
            train = txts[:int(num * 0.99)]
            test = txts[int(num * 0.99):]
            self.train_list, self.test_list = train, test
            self.train_offset=0
            self.test_offset=0
        else:
            self.test_texts = txts
            self.offset = 0

    def preprocess(self, tokens, txts):
        x = []
        y = []
        new = []
        for token, txt in zip(tokens, txts):
            # print(py,txt)
            if not self.check_valid(token, self.am_featurizer.vocab_array):
                logging.info('{} pinyin not all in token,continue'.format(txt))
                continue
            if not self.check_valid(txt, self.lm_featurizer.vocab_array):
                logging.info('{}  not all in token,continue'.format(txt))
                continue
            # try:
            x_ = [self.am_featurizer.startid()]
            y_ = [self.lm_featurizer.startid()]
            for i in token:
                x_.append(self.am_featurizer.token_to_index[i])
            for i in txt:
                y_.append(self.lm_featurizer.token_to_index[i])
            x_.append(self.am_featurizer.endid())
            y_.append(self.lm_featurizer.endid())
            x.append(np.array(x_))
            y.append(np.array(y_))
            new.append(txt)
        return x, y, new
    def only_chinese(self, word):
        txt = ''
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                txt += ch
            else:
                continue
        return txt
    def bert_decode(self, x, x2=None):
        tokens, segs = [], []
        if x2 is not None:
            for i, j in zip(x, x2):
                t, s = self.bert_token.encode(''.join(i))
                index = np.where(j == 2)[0]
                if len(index) > 0:
                    for n in index:
                        t[int(n)] = 103
                tokens.append(t)
                segs.append(s)
        else:
            for i in x:
                t, s = self.bert_token.encode(''.join(i))
                tokens.append(t)
                segs.append(s)
        return tokens, segs

    def pad(self, x, mode=1):
        length = 0

        for i in x:
            length = max(length, len(i))
        if mode == 2:
            for i in range(len(x)):
                pading = np.ones([length - len(x[i]), x[i].shape[1]]) * -10.
                x[i] = np.vstack((x[i], pading))

        else:
            x = pad_sequences(x, length, padding='post', truncating='post')
        return x

    def get_bert_feature(self, bert_t, bert_s):

        length = [len(i) for i in bert_t]
        max_len = max(length)
        bert_s = tf.keras.preprocessing.sequence.pad_sequences(bert_s, max_len, padding='post', truncating='post')
        bert_t = tf.keras.preprocessing.sequence.pad_sequences(bert_t, max_len, padding='post', truncating='post')
        features = self.bert.predict([bert_t, bert_s])

        for idx, l in enumerate(length):
            features[idx, l:] = -10.

        return features

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
        samples=[]
        x = []
        y = []

        for i in range(self.batch*10):
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
            txt=line.strip()
            txt=txt.replace(' ','')
            if self.config['only_chinese']:
                txt=self.only_chinese(txt)
            py=self.text_to_vocab(txt)
            if self.check_valid(py, self.am_featurizer.vocab_array) is not True:
                logging.info('{} pinyin  {} not  in token,skip'.format(txt,self.check_valid(py, self.am_featurizer.vocab_array)))
                continue
            if self.check_valid(txt, self.lm_featurizer.vocab_array) is not True:
                logging.info('{}  txt {} not in token,skip'.format(txt,self.check_valid(txt, self.lm_featurizer.vocab_array)))
                continue
            x_ = [self.am_featurizer.startid()]
            y_ = [self.lm_featurizer.startid()]
            for i in py:
                x_.append(self.am_featurizer.token_to_index[i])
            for i in txt:
                y_.append(self.lm_featurizer.token_to_index[i])
            x_.append(self.am_featurizer.endid())
            y_.append(self.lm_featurizer.endid())
            x.append(np.array(x_))
            y.append(np.array(y_))
            samples.append(txt)
            if len(samples)==self.batch:
                break
        e_bert_t, e_bert_s = self.bert_decode(samples)
        e_features = self.get_bert_feature(e_bert_t, e_bert_s)
        x = self.pad(x)
        y = self.pad(y)
        e_features = self.pad(e_features, 2)

        x = np.array(x)
        y = np.array(y)
        e_features = np.array(e_features, dtype='float32')

        return x, y, e_features

    def eval_generate(self, ):

        samples = []
        x = []
        y = []
        for i in range(self.batch * 10):
            line = self.test_texts[self.offset]
            self.offset += 1
            if self.offset > len(self.test_texts) - 1:
                self.offset = 0
            txt = line.strip()
            txt = txt.replace(' ', '')
            if self.config['only_chinese']:
                txt = self.only_chinese(txt)
            py = self.text_to_vocab(txt)
            if self.check_valid(py, self.am_featurizer.vocab_array) is not True:
                logging.info('{} pinyin  {} not  in token,skip'.format(txt,
                                                                self.check_valid(py, self.am_featurizer.vocab_array)))
                continue
            if self.check_valid(txt, self.lm_featurizer.vocab_array) is not True:
                logging.info('{}  txt {} not in token,skip'.format(txt, self.check_valid(txt, self.lm_featurizer.vocab_array)))
                continue
            x_ = [self.am_featurizer.startid()]
            y_ = [self.lm_featurizer.startid()]
            for i in py:
                x_.append(self.am_featurizer.token_to_index[i])
            for i in txt:
                y_.append(self.lm_featurizer.token_to_index[i])
            x_.append(self.am_featurizer.endid())
            y_.append(self.lm_featurizer.endid())
            x.append(np.array(x_))
            y.append(np.array(y_))
            samples.append(txt)
            if len(samples) == self.batch:
                break
        x = self.pad(x)
        y = self.pad(y)
        x = np.array(x, 'int32')
        y = np.array(y, 'int32')
        return x, y

    def generator(self, train=True):
        while 1:
            x, y, features = self.generate(train)
            if len(x) == 0:
                logging.info('load data length zero,continue')
                continue
            yield x, y, features
