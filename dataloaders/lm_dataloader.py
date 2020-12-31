from utils.text_featurizers import TextFeaturizer
import pypinyin
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras_bert import Tokenizer, load_vocabulary, load_trained_model_from_checkpoint
import random
import tensorflow as tf
import os
class LM_DataLoader():
    def __init__(self, config,training=True):
        self.train = training
        self.init_all(config)

        self.vocab_featurizer = TextFeaturizer(config['lm_vocab'])
        self.word_featurizer = TextFeaturizer(config['lm_word'])
        self.init_text_to_vocab()
        self.batch = config['running_config']['batch_size']
        self.epochs=1
    def init_bert(self, config, checkpoint):
        model = load_trained_model_from_checkpoint(config, checkpoint, trainable=False, seq_len=None)
        return model
    def load_state(self,outdir):
        try:
            self.train_pick=np.load(os.path.join(outdir,'dg_state.npy')).flatten().tolist()
            self.epochs=1+int(np.mean(self.train_pick))
        except FileNotFoundError:
            print('not found state file')
        except:
            print('load state falied,use init state')
    def save_state(self,outdir):
        np.save(os.path.join(outdir,'dg_state.npy'),np.array(self.train_pick))
    def return_data_types(self):


        return  (tf.int32, tf.int32, tf.float32)
    def return_data_shape(self):


        return (
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None,768])
            )
    def get_per_epoch_steps(self):
        return len(self.train_texts)//self.batch

    def eval_per_epoch_steps(self):
        return len(self.test_texts) // self.batch
    def init_all(self, config):
        if self.train:
            bert_config = config['bert']['config_json']
            bert_checkpoint =config['bert']['bert_ckpt']
            bert_vocab =config['bert']['bert_vocab']
            bert_vocabs = load_vocabulary(bert_vocab)
            self.bert_token = Tokenizer(bert_vocabs)
            self.bert = self.init_bert(bert_config, bert_checkpoint)
        self.get_sentence(config['train_list'] if self.train else config['eval_list'],training=self.train)

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
            return pypinyin.lazy_pinyin(txt, 1, errors='ignore')

        self.text_to_vocab = text_to_vocab_func

    def get_sentence(self, data_path,training):
        from tqdm import tqdm

        with open(data_path, encoding='utf-8') as f:
            data = f.readlines()

        txts = []
        for txt in tqdm(data):
            txt = txt.strip()
            if len(txt) < 150 and self.check_valid(txt,self.word_featurizer.vocab_array):
                continue
            txts.append(txt)
        if training:
            num=len(txts)
            train=txts[:int(num*0.99)]
            test=txts[int(num*0.99):]
            self.train_texts, self.test_texts=train,test
            self.train_pick = [0] * len(self.train_texts)
        else:
            self.test_texts=txts
            self.offset=0

    def preprocess(self, tokens, txts):
        x = []
        y = []
        for token, txt in zip(tokens, txts):
            # print(py,txt)
            # try:
            x_ = [self.vocab_featurizer.startid()]
            y_ = [self.word_featurizer.startid()]
            for i in token:
                x_.append(self.vocab_featurizer.token_to_index[i])
            for i in txt:
                y_.append(self.word_featurizer.token_to_index[i])
            x_.append(self.vocab_featurizer.endid())
            y_.append(self.word_featurizer.endid())
            x.append(np.array(x_))
            y.append(np.array(y_))

        return x, y

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
        f = []
        for t, s in zip(bert_t, bert_s):
            t = np.expand_dims(np.array(t), 0)
            s = np.expand_dims(np.array(s), 0)
            feature = self.bert.predict([t, s])
            f.append(feature[0])
        return f
    def check_valid(self,txt,vocab_list):
        if len(txt)==0:
            return False
        for n in txt:
            if n in vocab_list:
                pass
            else:
                return False
        return True
    def generate(self,train=True):
        if train:
            indexs = np.argsort(self.train_pick)[:2 * self.batch]
            indexs = random.sample(indexs.tolist(), self.batch)
            sample = [self.train_texts[i] for i in indexs]
            for i in indexs:
                self.train_pick[int(i)] += 1
            self.epochs = 1+int(np.mean(self.train_pick))
        else:
            sample = random.sample(self.test_texts, self.batch)
        trainx = [self.text_to_vocab(i) for i in sample]
        trainy = sample
        x, y = self.preprocess(trainx, trainy)
        e_bert_t, e_bert_s = self.bert_decode(trainy)
        e_features = self.get_bert_feature(e_bert_t, e_bert_s)
        x = self.pad(x)
        y = self.pad(y)
        e_features = self.pad(e_features, 2)

        x = np.array(x)
        y = np.array(y)
        e_features = np.array(e_features, dtype='float32')

        return x, y, e_features

    def eval_generate(self,):


        sample = self.test_texts[self.offset:self.offset+self.batch]
        self.offset+=self.batch
        trainx = [self.text_to_vocab(i) for i in sample]
        trainy = sample
        x, y = self.preprocess(trainx, trainy)
        x = self.pad(x)
        y = self.pad(y)
        x = np.array(x,'int32')
        y = np.array(y,'int32')
        return x, y
    def generator(self,train=True):
        while 1:
            x, y, features=self.generate(train)
            yield x,y,features

