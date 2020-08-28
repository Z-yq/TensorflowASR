from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer
import numpy as np
from augmentations.augments import Augmentation
import random
import tensorflow as tf
from jieba.posseg import lcut
from keras_bert import Tokenizer, load_vocabulary, load_trained_model_from_checkpoint
class MultiTask_DataLoader():

    def __init__(self, config_dict,training=True):
        self.speech_config = config_dict['speech_config']


        self.text1_config = config_dict['decoder1_config']
        self.text2_config = config_dict['decoder2_config']
        self.text3_config = config_dict['decoder3_config']
        self.text4_config = config_dict['decoder4_config']
        self.augment_config = config_dict['augments_config']

        self.batch = config_dict['learning_config']['running_config']['batch_size']
        self.speech_featurizer = SpeechFeaturizer(self.speech_config)
        self.token1_featurizer = TextFeaturizer(self.text1_config)
        self.token2_featurizer = TextFeaturizer(self.text2_config)
        self.token3_featurizer = TextFeaturizer(self.text3_config)
        self.token4_featurizer = TextFeaturizer(self.text4_config)
        self.make_file_list(self.speech_config['train_list'] if training else self.speech_config['eval_list'],training)
        self.make_maps(config_dict)
        self.augment = Augmentation(self.augment_config)
        self.epochs = 1
        self.LAS=True
        self.steps = 0
        self.init_bert(config_dict)
    def load_bert(self, config, checkpoint):
        model = load_trained_model_from_checkpoint(config, checkpoint, trainable=False, seq_len=None)
        return model

    def init_bert(self,config):
        bert_config = config['bert']['config_json']
        bert_checkpoint = config['bert']['bert_ckpt']
        bert_vocab = config['bert']['bert_vocab']
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

        return (tf.float32, tf.float32, tf.float32,tf.int32, tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32, tf.int32,tf.float32)

    def return_data_shape(self):
        f,c=self.speech_featurizer.compute_feature_dim()

        return (
            tf.TensorShape([None,None,f,c]),
            tf.TensorShape([None,None,1]),
            tf.TensorShape([None, None, 768]),
            tf.TensorShape([None,]),
            tf.TensorShape([None,None]),
            tf.TensorShape([None,]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None,None,None])
        )

    def get_per_epoch_steps(self):
        return len(self.train_list)//self.batch
    def eval_per_epoch_steps(self):
        return len(self.test_list)//self.batch
    def make_maps(self,config):
        with open(config['map_path']['pinyin'],encoding='utf-8') as f:
            data=f.readlines()
        data=[i.strip() for i in data if i!='']
        self.py_map={}
        for line in data:
            key,py=line.strip().split('\t')
            self.py_map[key]=py
            if len(py.split(' '))>1:
                for i,j in zip(list(key),py.split(' ')):
                    self.py_map[i]=j
        with open(config['map_path']['phone'],encoding='utf-8') as f:
            data=f.readlines()
        data=[i.strip() for i in data if i!='']
        self.phone_map={}
        phone_map={}
        for line in data:
            key,py=line.strip().split('\t')
            phone_map[key]=py
        for key in self.py_map.keys():
            key_py=self.py_map[key]
            if len(key)>1:
                phone=[]
                for n in key_py.split(' '):
                    phone+=[phone_map[n]]
                self.phone_map[key]=' '.join(phone)
            else:
                self.phone_map[key]=phone_map[self.py_map[key]]
    def map(self,txt):
        cut=lcut(txt)
        pys=[]
        phones=[]
        words=[]
        for i in cut:
            word=i.word
            if word in self.py_map.keys():
                py=self.py_map[word]
                phone=self.phone_map[word]
                pys+=py.split(' ')
                phones+=phone.split(' ')
                words+=list(''.join(py.split(' ')))
            else:
                for j in word:
                    pys+=[self.py_map[j]]
                    phones+=self.phone_map[j].split(' ')
                    words+=list(''.join(self.py_map[j]))
        return pys,phones,words

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
                pad = np.ones([max_input - mels[i].shape[0], mels[i].shape[1],mels[i].shape[2]]) * mels[i].min()
                mels[i] = np.vstack((mels[i], pad))

        wavs_ = self.speech_featurizer.pad_signal(wavs_, max_wav)

        x = np.array(mels, 'float32')
        label_ = np.array(label_, 'int32')

        input_length = np.array(input_length, 'int32')
        label_length_ = np.array(label_length_, 'int32')

        wavs_ = np.array(np.expand_dims(wavs_, -1), 'float32')

        return x, wavs_, input_length, label_, label_length_

    def make_file_list(self, wav_list,training=True):
        with open(wav_list, encoding='utf-8') as f:
            data = f.readlines()
        data=[i.strip()  for i in data if i!='']
        num = len(data)
        if training:
            self.train_list = data[:int(num * 0.99)]
            self.test_list = data[int(num * 0.99):]
            np.random.shuffle(self.train_list)
            self.pick_index = [0.] * len(self.train_list)
        else:
            self.test_list=data
            self.offset=0
    def only_chinese(self, word):

        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                pass
            else:
                return False

        return True
    def eval_data_generator(self):
        sample=self.test_list[self.offset:self.offset+self.batch]
        self.offset+=self.batch
        mels = []
        input_length = []

        words_label = []
        words_label_length = []

        phone_label = []
        phone_label_length = []

        py_label = []
        py_label_length = []

        txt_label = []
        txt_label_length = []
        
        bert_features=[]
        wavs = []

        max_wav = 0
        max_input = 0
        max_label_words = 0
        max_label_phone = 0
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

            if not self.only_chinese(txt):
                continue

            speech_feature = self.speech_featurizer.extract(data)
            max_input = max(max_input, speech_feature.shape[0])

            py,phone,word = self.map(txt)
            if len(py) == 0:
                continue
            e_bert_t, e_bert_s = self.bert_decode([txt])
            bert_feature = self.get_bert_feature(e_bert_t, e_bert_s)

            word_text_feature = self.token1_featurizer.extract(word)
            phone_text_feature = self.token2_featurizer.extract(phone)
            py_text_feature = self.token3_featurizer.extract(py)
            txt_text_feature = self.token4_featurizer.extract(list(txt))
            max_label_words = max(max_label_words, len(word_text_feature))
            max_label_phone = max(max_label_phone, len(phone_text_feature))
            max_label_py = max(max_label_py, len(py_text_feature))
            max_label_txt = max(max_label_txt, len(txt_text_feature))
        
            max_wav = max(max_wav, len(data))
            if speech_feature.shape[0] / self.speech_config['reduction_factor'] < len(py_text_feature):
                continue
            mels.append(speech_feature)
            wavs.append(data)
            input_length.append(speech_feature.shape[0] // self.speech_config['reduction_factor'])
            words_label.append(np.array(word_text_feature))
            words_label_length.append(len(word_text_feature))

            phone_label.append(np.array(phone_text_feature))
            phone_label_length.append(len(phone_text_feature))

            py_label.append(np.array(py_text_feature))
            py_label_length.append(len(py_text_feature))

            txt_label.append(np.array(txt_text_feature))
            txt_label_length.append(len(txt_text_feature))
            bert_features.append(bert_feature)

        for i in range(len(mels)):
            if mels[i].shape[0] < max_input:
                pad = np.ones([max_input - mels[i].shape[0], mels[i].shape[1], mels[i].shape[2]]) * mels[i].min()
                mels[i] = np.vstack((mels[i], pad))

        for i in range(len(bert_features)):

            if bert_features[i].shape[0] < max_label_txt:
                pading = np.ones([max_label_txt - len(bert_features[i]), 768]) * -10.
                bert_features[i] = np.vstack((bert_features[i], pading))


        wavs = self.speech_featurizer.pad_signal(wavs, max_wav)
        words_label = self.pad(words_label, max_label_words)
        phone_label = self.pad(phone_label, max_label_phone)
        py_label = self.pad(py_label, max_label_py)
        txt_label = self.pad(txt_label, max_label_txt)

        x = np.array(mels, 'float32')
        bert_features = np.array(bert_features, 'float32')
        words_label = np.array(words_label, 'int32')
        phone_label = np.array(phone_label, 'int32')
        py_label = np.array(py_label, 'int32')
        txt_label = np.array(txt_label, 'int32')

        input_length = np.array(input_length, 'int32')
        words_label_length = np.array(words_label_length, 'int32')
        phone_label_length = np.array(phone_label_length, 'int32')
        py_label_length = np.array(py_label_length, 'int32')
        txt_label_length = np.array(txt_label_length, 'int32')

        wavs = np.array(np.expand_dims(wavs, -1), 'float32')

        return x, wavs, bert_features,input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length, txt_label, txt_label_length
    def pad(self,words_label,max_label_words):
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

        if train:
            batch=self.batch if self.augment.available() else self.batch*2
            indexs = np.argsort(self.pick_index)[:batch]
            indexs = random.sample(indexs.tolist(), batch//2)
            sample = [self.train_list[i] for i in indexs]
            for i in indexs:
                self.pick_index[int(i)] += 1
            self.epochs = int(np.mean(self.pick_index))
        else:
            sample = random.sample(self.test_list, self.batch)

        mels = []
        input_length = []

        words_label = []
        words_label_length = []

        phone_label = []
        phone_label_length = []

        py_label = []
        py_label_length = []

        txt_label = []
        txt_label_length = []

        bert_features = []
        wavs = []

        max_wav = 0
        max_input = 0
        max_label_words = 0
        max_label_phone = 0
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

            if not self.only_chinese(txt):
                continue

            speech_feature = self.speech_featurizer.extract(data)


            py, phone, word = self.map(txt)
            if len(py) == 0 or len(phone)==0 or len(word)==0:
                continue
            e_bert_t, e_bert_s = self.bert_decode([txt])
            bert_feature = self.get_bert_feature(e_bert_t, e_bert_s)

            word_text_feature = self.token1_featurizer.extract(word)
            phone_text_feature = self.token2_featurizer.extract(phone)
            py_text_feature = self.token3_featurizer.extract(py)
            txt_text_feature = self.token4_featurizer.extract(list(txt))

            if speech_feature.shape[0] / self.speech_config['reduction_factor'] < len(py_text_feature) or \
                    speech_feature.shape[0] / self.speech_config['reduction_factor'] < len(word_text_feature) or \
                    speech_feature.shape[0] / self.speech_config['reduction_factor'] < len(phone_text_feature):
                continue
            max_input = max(max_input, speech_feature.shape[0])
            max_label_words = max(max_label_words, len(word_text_feature))
            max_label_phone = max(max_label_phone, len(phone_text_feature))
            max_label_py = max(max_label_py, len(py_text_feature))
            max_label_txt = max(max_label_txt, len(txt_text_feature))

            max_wav = max(max_wav, len(data))
            mels.append(speech_feature)
            wavs.append(data)
            input_length.append(speech_feature.shape[0] // self.speech_config['reduction_factor'])
            words_label.append(np.array(word_text_feature))
            words_label_length.append(len(word_text_feature))

            phone_label.append(np.array(phone_text_feature))
            phone_label_length.append(len(phone_text_feature))

            py_label.append(np.array(py_text_feature))
            py_label_length.append(len(py_text_feature))

            txt_label.append(np.array(txt_text_feature))
            txt_label_length.append(len(txt_text_feature))
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

                if not self.only_chinese(txt):
                    continue
                data=self.augment.process(data)
                speech_feature = self.speech_featurizer.extract(data)


                py, phone, word = self.map(txt)
                if len(py) == 0 or len(phone) == 0 or len(word) == 0:
                    continue
                e_bert_t, e_bert_s = self.bert_decode([txt])
                bert_feature = self.get_bert_feature(e_bert_t, e_bert_s)

                word_text_feature = self.token1_featurizer.extract(word)
                phone_text_feature = self.token2_featurizer.extract(phone)
                py_text_feature = self.token3_featurizer.extract(py)
                txt_text_feature = self.token4_featurizer.extract(list(txt))



                if speech_feature.shape[0] / self.speech_config['reduction_factor'] < len(py_text_feature) or \
                        speech_feature.shape[0] / self.speech_config['reduction_factor'] < len(word_text_feature) or \
                        speech_feature.shape[0] / self.speech_config['reduction_factor'] < len(phone_text_feature):
                    continue
                max_input = max(max_input, speech_feature.shape[0])
                max_wav = max(max_wav, len(data))
                max_label_words = max(max_label_words, len(word_text_feature))
                max_label_phone = max(max_label_phone, len(phone_text_feature))
                max_label_py = max(max_label_py, len(py_text_feature))
                max_label_txt = max(max_label_txt, len(txt_text_feature))
                mels.append(speech_feature)
                wavs.append(data)
                input_length.append(speech_feature.shape[0] // self.speech_config['reduction_factor'])
                words_label.append(np.array(word_text_feature))
                words_label_length.append(len(word_text_feature))

                phone_label.append(np.array(phone_text_feature))
                phone_label_length.append(len(phone_text_feature))

                py_label.append(np.array(py_text_feature))
                py_label_length.append(len(py_text_feature))

                txt_label.append(np.array(txt_text_feature))
                txt_label_length.append(len(txt_text_feature))
                bert_features.append(bert_feature)

        for i in range(len(mels)):
            if mels[i].shape[0] < max_input:
                pad = np.ones([max_input - mels[i].shape[0], mels[i].shape[1], mels[i].shape[2]]) * mels[i].min()
                mels[i] = np.vstack((mels[i], pad))
        for i in range(len(bert_features)):
            if bert_features[i].shape[0]<max_label_txt:
                pading = np.ones([max_label_txt - len(bert_features[i]), 768]) * -10.
                bert_features[i] = np.vstack((bert_features[i], pading))

        wavs = self.speech_featurizer.pad_signal(wavs, max_wav)
        words_label = self.pad(words_label, max_label_words)
        phone_label = self.pad(phone_label, max_label_phone)
        py_label = self.pad(py_label, max_label_py)
        txt_label = self.pad(txt_label, max_label_txt)

        x = np.array(mels, 'float32')
        bert_features = np.array(bert_features, 'float32')
        words_label = np.array(words_label, 'int32')
        phone_label = np.array(phone_label, 'int32')
        py_label = np.array(py_label, 'int32')
        txt_label = np.array(txt_label, 'int32')

        input_length = np.array(input_length, 'int32')
        words_label_length = np.array(words_label_length, 'int32')
        phone_label_length = np.array(phone_label_length, 'int32')
        py_label_length = np.array(py_label_length, 'int32')
        txt_label_length = np.array(txt_label_length, 'int32')

        wavs = np.array(np.expand_dims(wavs, -1), 'float32')

        return x, wavs, bert_features,input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length, txt_label, txt_label_length
    def generator(self,train=True):
        while 1:
            x, wavs,bert_feature, input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length, txt_label, txt_label_length=self.generate(train)

            guide_matrix = self.guided_attention(input_length, txt_label_length, np.max(input_length),
                                                 txt_label_length.max())
            yield x, wavs, bert_feature,input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length, txt_label, txt_label_length,guide_matrix


if __name__ == '__main__':
    from utils.user_config import UserConfig
    import tensorflow as tf
    config = UserConfig(r'D:\TF2-ASR\configs\am_data.yml',r'D:\TF2-ASR\configs\MultiConformer.yml')
    config['decoder1_config']['model_type']='LAS'
    config['decoder2_config']['model_type']='LAS'
    config['decoder3_config']['model_type']='LAS'
    config['decoder4_config']['model_type']='LAS'
    dg = MultiTask_DataLoader(config)
    datasets=tf.data.Dataset.from_generator(dg.generator,dg.return_data_types(),dg.return_data_shape())

    # x, wavs, input_length, y1, label_length1 = dg.generator()
    # print(x.shape, wavs.shape, input_length.shape, y1.shape, label_length1.shape)
