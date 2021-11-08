import random
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras_bert import Tokenizer, load_vocabulary, load_trained_model_from_checkpoint
import logging
from utils.text_featurizers import TextFeaturizer
import os


class Punc_DataLoader():
    def __init__(self, config, training=True):
        self.train = training
        self.init_all(config)
        self.running_config=config['running_config']
        self.vocab_featurizer = TextFeaturizer(config['punc_vocab'])
        self.bd_featurizer = TextFeaturizer(config['punc_biaodian'])
        self.bd = self.bd_featurizer.vocab_array
        self.batch = config['running_config']['batch_size']
        self.epochs = 1

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
            tf.TensorShape([self.batch, None]),
            tf.TensorShape([self.batch, None]),
            tf.TensorShape([self.batch, None, 768])
        )


    def eval_return_data_types(self):

        return (tf.int32, tf.int32)

    def eval_return_data_shape(self):

        return (
            tf.TensorShape([self.batch, None]),
            tf.TensorShape([self.batch, None]),
        )

    def get_per_epoch_steps(self):
        return len(self.train_list) // self.batch

    def eval_per_epoch_steps(self):
        return len(self.test_list) // self.batch

    def init_all(self, config):
        if self.train:
            bert_config = config['bert']['config_json']
            bert_checkpoint = config['bert']['bert_ckpt']
            bert_vocab = config['bert']['bert_vocab']
            bert_vocabs = load_vocabulary(bert_vocab)
            self.bert_token = Tokenizer(bert_vocabs)
            self.bert = self.init_bert(bert_config, bert_checkpoint)
        self.get_sentence(training=self.train)

    def get_sentence(self,  training):
        train_list = self.running_config['train_list']
        test_list = self.running_config['eval_list']
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

    def preprocess(self,  txts):
        x = []
        for  txt in  txts:
            x_ = [self.vocab_featurizer.startid()]
            for i in txt:
                x_.append(self.vocab_featurizer.token_to_index[i])
            x_.append(self.vocab_featurizer.endid())
            x.append(np.array(x_))
        return x

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
        elif mode == 3:
            for i in range(len(x)):
                pading = np.zeros([length - len(x[i]), x[i].shape[1]])
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

    def get_target(self, text):

        bd = self.bd
        zh = []
        bd_ = [[self.bd_featurizer.pad]]
        for n in text:
            if n in bd:
                bd_[-1].append(bd.index(n))
            else:
                zh.append(n)
                bd_.append([self.bd_featurizer.pad])
        zh_txt=''.join(zh)
        bd_txt=bd_+[[self.bd_featurizer.pad]]
        return zh_txt, bd_txt

    def process_punc(self, puncs):
        x = []
        for punc in puncs:
            x_ = []
            for i in range(len(punc)):
                if len(punc[i]) == 1:
                    x_+= [1]
                else:
                    x_+= punc[i][-1:]
            x.append(np.array(x_,'int32'))
        return x

    def check_valid(self, txt, vocab_list):
        if len(txt) == 0:
            return False
        for n in txt:
            if n in vocab_list:
                pass
            else:
                return n
        return True
    def generate(self,train):

        trainx=[]
        trainy=[]
        for i in range(self.batch * 10):
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

            line = line.strip()
            if len(line)<30:
                extra=random.sample(self.train_list, 1)[0]
                extra = extra.strip()
                line+=extra

            if self.check_valid(line,self.vocab_featurizer.vocab_array+self.bd) is not True:
                continue
            try:
                x,y=self.get_target(line)
            except:
                continue
            trainx.append(x)
            trainy.append(y)
            if len(trainx)==self.batch:
                break

        inp_tokens= self.preprocess(trainx)
        e_bert_t, e_bert_s = self.bert_decode(trainx)
        e_features = self.get_bert_feature(e_bert_t, e_bert_s)
        trainy=self.process_punc(trainy)
        inp_tokens = self.pad(inp_tokens)
        trainy = self.pad(trainy)
        e_features = self.pad(e_features, 2)
        inp_tokens = np.array(inp_tokens)
        trainy = np.array(trainy)
        e_features = np.array(e_features, dtype='float32')

        return inp_tokens, trainy, e_features

    def eval_generate(self):
        trainx = []
        trainy = []
        for i in range(self.batch * 10):

            line = self.test_list[self.test_offset]
            self.test_offset += 1
            if self.test_offset > len(self.test_list) - 1:
                self.test_offset = 0

            line = line.strip()
            if len(line) < 30:
                extra = random.sample(self.train_list, 1)[0]
                extra = extra.strip()
                line += extra

            if self.check_valid(line, self.vocab_featurizer.vocab_array + self.bd) is not True:
                continue
            try:
                x, y = self.get_target(line)
            except:
                continue
            trainx.append(x)
            trainy.append(y)
            if len(trainx) == self.batch:
                break

        inp_tokens = self.preprocess(trainx)
        trainy = self.process_punc(trainy)
        inp_tokens = self.pad(inp_tokens)
        trainy = self.pad(trainy)

        inp_tokens = np.array(inp_tokens)
        trainy = np.array(trainy)


        return inp_tokens, trainy,
    def generator(self,train=True):
        while 1:
            x, y, features = self.generate(train)
            if x.shape[1]!=y.shape[1] and y.shape[1] !=features.shape[1]:
                logging.info('bad batch,skip')
                continue
            yield x, y, features


    def eval_generator(self):
        while 1:
            x, y, = self.eval_generate()
            if x.shape[1]!=y.shape[1]:
                logging.info('bad batch,skip')
                continue
            yield x, y