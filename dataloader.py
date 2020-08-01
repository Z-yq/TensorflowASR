import tensorflow as tf
import hparams
import librosa
import os
from utils.token_tool import MakeS2SDict
import random
import pypinyin
from utils import audio
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras_bert import Tokenizer,load_vocabulary,load_trained_model_from_checkpoint

class AM_DataLoader():

    def __init__(self, itokens1, itokens2, itokens3, itokens4, batch=32, SampleRate=8000, n_mels=64,noise_path= './noise/',train_list='./train_list.txt'
                ):
        self.n_mels = n_mels
        self.Sr = SampleRate
        self.batch = batch

        self.make_file_list(train_list)


        self.epochs = 1

        self.steps = 0

        self.itokens3 = itokens3
        self.itokens1 = itokens1
        self.itokens2 = itokens2
        self.itokens4 = itokens4



        noise = os.listdir(noise_path)
        self.noise = []
        for i in noise:
            n_fp = os.path.join(noise_path, i)
            n_wav, _ = librosa.load(n_fp, sr=hparams.sample_rate)
            data = None
            inver = librosa.effects.split(n_wav, 25)
            for s, e in inver:
                if data is None:
                    data = n_wav[s:e]
                else:
                    data = np.hstack((data, n_wav[s:e]))
            n_wav = data
            n_wav = n_wav / np.abs(n_wav).max()
            self.noise.append(n_wav)
        print('total noise', len(self.noise))
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
        # logging.info('Dict loaded,mask frequence %.2f' % self.freq)

    def Add_noise(self, x, d, SNR):
        P_signal = np.sum(abs(x) ** 2)

        P_d = np.sum(abs(d) ** 2)

        P_noise = P_signal / 10 ** (SNR / 10)

        noise = np.sqrt(P_noise / P_d) * d
        num = min(x.shape[0], noise.shape[0])
        if num < noise.shape[0]:
            pick_num = np.random.randint(0, noise.shape[0] - num)
            noise = noise[int(pick_num):int(pick_num) + num]
        noise_signal = x[:num] + noise[:num]
        return noise_signal

    def get_freq(self):

        for i in self.file_dict.keys():
            print(i, len(self.file_dict[i]))

    def remake(self, pre):
        new = []
        for i in pre:
            y = [2]
            for j in i:
                if j != -1 and j != 0:
                    y.append(j)
            y.append(3)
            y = np.array(y)
            new.append(y)
        new = np.array(new)
        return new


    def make_file_list(self,wav_list):
        with open(wav_list, encoding='utf-8') as f:
            data=f.readlines()
        num=len(data)
        self.train_list=data[:int(num*0.99)]
        self.test_list = data[int(num * 0.99):]
        np.random.shuffle(self.train_list)
        self.pick_index = [0.] * len(self.train_list)


    def add_mask(self, mel, mode=1):
        if np.random.random() > 0.5:
            mask_value = np.random.random()
        else:
            mask_value = np.random.random() * -1
        if mode == 1:
            mask = np.ones_like(mel)
            keys = list(range(mask.shape[1]))
            key = random.sample(keys, int(len(keys) * 0.2))
            for i in key:
                mask[:, i] *= 0
            masked = mel * mask
            masked += np.where(mask == 0, mask_value, 0.)
        elif mode == 2:
            mask = np.ones_like(mel)
            keys = list(range(mask.shape[0]))
            key = random.sample(keys, int(len(keys) * 0.2))
            for i in key:
                mask[i] *= 0
            masked = mel * mask
            masked += np.where(mask == 0, mask_value, 0.)
        elif mode == 0:
            masked = mel.copy()
            arr = np.random.random([mel.shape[0], mel.shape[1]])
            mask = arr < 0.15
            masked[mask] = mask_value
        else:
            masked = mel.copy()
            arr = np.random.random([mel.shape[0], mel.shape[1]])
            mask = arr < 0.15
            masked[mask] = mask_value
        return masked

    def mask_mel(self, mel):
        mode = random.sample(list(np.repeat([0, 1, 2], 10)), 1)[0]
        mel = self.add_mask(mel, mode=mode)
        return mel

    def add_noise(self, wavs):
        wavs_ = []
        mels = []
        for wav in wavs:

            wav = wav[:, 0]
            if np.random.random() < 0.5:
                n_num = np.random.randint(0, len(self.noise))
                n_wav = self.noise[n_num]
                while len(wav) + 20 > len(n_wav):
                    n_wav = np.hstack((n_wav, n_wav))
                start = np.random.randint(0, len(n_wav) - len(wav) - 10)
                n_wav = n_wav[start:start + len(wav)]
                SNR = np.random.randint(-5, 5)
                wav = self.Add_noise(wav, n_wav, SNR)
            else:

                mask = np.random.random(len(wav))
                wav = wav * np.where(mask < 0.25, 0., 1.)
            wav = audio.preemphasis(wav, 0.97)
            wav /= np.abs(wav).max()
            mel = audio.melspectrogram(wav, hparams).T

            # print(mel.shape)
            if len(wav) % hparams.hop_size == 0:
                mel = mel[:-1]
            wavs_.append(wav[:, np.newaxis])
            mels.append(mel)
        mels = np.array(mels, dtype='float32')
        wavs_ = np.array(wavs_, dtype='float32')

        return mels, wavs_

    def only_chinese(self,word):

        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                pass
            else:
                return False

        return True
    def generator(self, train=True):

        if train:
            indexs = np.argsort(self.pick_index)[:2 * self.batch]
            indexs = random.sample(indexs.tolist(), self.batch)
            sample = [self.train_list[i] for i in indexs]
            for i in indexs:
                self.pick_index[int(i)] += 1
            self.epochs = np.mean(self.pick_index)
        else:
            sample = random.sample(self.test_list, self.batch)



        mels = []
        input_length = []

        y1 = []
        label_length1 = []

        y2 = []
        label_length2 = []

        y3 = []
        label_length3 = []

        y4 = []
        label_length4 = []

        wavs = []

        max_wav = 0
        max_input = 0
        max_label1 = 0
        max_label2 = 0
        max_label3 = 0
        max_label4 = 0
        for i in sample:
            wp,txt = i.strip().split('\t')
            try:
                data, sr = librosa.load(wp, sr=self.Sr)
            except:
                continue
            if len(data) < 400:
                continue
            elif len(data) > self.Sr * 15:
                continue

            if not self.only_chinese(txt):
                continue
            data = librosa.effects.trim(data, 20)[0]
            data = audio.preemphasis(data, 0.97)
            data /= np.abs(data).max()
            mel = audio.melspectrogram(data, hparams)
            mel = mel.transpose((1, 0))
            if len(data) % hparams.hop_size == 0:
                mel = mel[:-1]
            if mel.shape[0] > max_input:
                max_input = mel.shape[0]
            py1 = pypinyin.lazy_pinyin(txt, 8, errors='ignore')
            py2_1 = pypinyin.lazy_pinyin(txt, 3, errors='ignore')
            py2_2 = pypinyin.lazy_pinyin(txt, 9, errors='ignore')
            py2 = []
            for p1, p2 in zip(py2_1, py2_2):
                if p1 != '':
                    py2.append(p1)
                if p2 != '':
                    py2.append(p2)
            py3 = pypinyin.pinyin(txt, errors='ignore')
            if len(py3) == 0:
                continue
            y_1 = []
            t_1 = ''
            for j in py1:
                t_1 += j
            for j in t_1:
                y_1.append(self.itokens1.id(j))
            y_1 = np.array(y_1)
            if len(y_1) == 0:
                continue
            y_2 = []
            # print(py2)
            for j in py2:
                y_2.append(self.itokens2.id(j))
            y_2 = np.array(y_2)
            # print(y_2)
            if len(y_2) == 0:
                continue
            y_3 = []
            for j in py3:
                y_3.append(self.itokens3.id(j[0]))
            if len(y_3) == 0:
                continue
            y_4 = []
            for j in txt:
                y_4.append(self.itokens4.id(j))

            y_4 = np.array(y_4)
            y_3 = np.array(y_3)
            if len(y_1) > max_label1:
                max_label1 = len(y_1)
            if len(y_2) > max_label2:
                max_label2 = len(y_2)
            if len(y_3) > max_label3:
                max_label3 = len(y_3)

            max_label4 = max(len(y_4), max_label4)
            max_wav = max(max_wav, len(data))
            if mel.shape[0] / 2 < len(y_3) or mel.shape[0] / 2 < len(y_1) or mel.shape[0] / 2 < len(y_2) or mel.shape[
                0]/2 < len(y_4):
                continue
            mels.append(mel)
            wavs.append(data)
            # wavs_grad.append(np.gradient(data))
            input_length.append(mel.shape[0])
            y1.append(y_1)
            label_length1.append(len(y_1))
            y2.append(y_2)
            label_length2.append(len(y_2))
            y3.append(y_3)
            label_length3.append(len(y_3))
            y4.append(y_4)
            label_length4.append(len(y_4))
            # except:
            #     continue

        for i in range(len(mels)):
            if mels[i].shape[0] < max_input:
                pad = np.ones([max_input - mels[i].shape[0], mels[i].shape[1]]) * -hparams.max_abs_value
                mels[i] = np.vstack((mels[i], pad))

        wavs = tf.keras.preprocessing.sequence.pad_sequences(wavs, max_wav, 'float32', 'post', 'post')
        for i in range(len(y1)):
            if y1[i].shape[0] < max_label1:
                pad = np.zeros(max_label1 - y1[i].shape[0])
                y1[i] = np.hstack((y1[i], pad))

        for i in range(len(y2)):
            if y2[i].shape[0] < max_label2:
                pad = np.zeros(max_label2 - y2[i].shape[0])
                y2[i] = np.hstack((y2[i], pad))

        for i in range(len(y3)):
            if y3[i].shape[0] < max_label3:
                pad = np.zeros(max_label3 - y3[i].shape[0])
                y3[i] = np.hstack((y3[i], pad))
        for i in range(len(y4)):
            if y4[i].shape[0] < max_label4:
                pad = np.zeros(max_label4 - y4[i].shape[0])
                y4[i] = np.hstack((y4[i], pad))

        x = np.array(mels,'float32')
        y1 = np.array(y1,'int32')
        y2 = np.array(y2,'int32')
        y3 = np.array(y3,'int32')
        y4 = np.array(y4,'int32')
        input_length = np.array(input_length,'int32')
        label_length1 = np.array(label_length1,'int32')
        label_length2 = np.array(label_length2,'int32')
        label_length3 = np.array(label_length3,'int32')
        label_length4 = np.array(label_length4,'int32')
        wavs = np.array(np.expand_dims(wavs, -1),'float32')

        return x ,wavs, input_length, y1, label_length1, y2, label_length2, y3, label_length3, y4, label_length4

class LM_DataLoader():
    def __init__(self,data_path):
        self.init_all(data_path)

    def init_bert(self,config,checkpoint):
        model=load_trained_model_from_checkpoint(config,checkpoint,trainable=False,seq_len=None)
        return model
    def init_all(self,data_path):
        config = './LMmodel/bert/bert_config.json'
        checkpoint = './LMmodel/bert/bert_model.ckpt'
        vocab = './LMmodel/bert/vocab.txt'
        self.pinyin_tokens, self.hans_tokens = MakeS2SDict(None, delimiter=' ', dict_file='./LMmodel/lm_tokens.txt')

        vocabs = load_vocabulary(vocab)
        self.bert_token = Tokenizer(vocabs)

        self.bert = self.init_bert(config, checkpoint)
        self.texts = self.get_sentence(data_path)
        # self.pick_prob=np.zeros(len(self.pinyins))


    def get_sentence(self,data_path):
        from tqdm import tqdm
        with open(data_path, encoding='utf-8') as f:
            data = f.readlines()

        txts=[]
        for txt in tqdm(data):
            txt=txt.strip()
            if len(txt)>150:
                continue
            txts.append(txt)

        return txts

    def preprocess(self,pinyins,txts):
        x=[]
        y=[]
        for py,txt in zip(pinyins,txts):
            # print(py,txt)
            try:
                x_=[self.pinyin_tokens.startid()]
                y_=[self.hans_tokens.startid()]
                for i in py:
                    x_.append(self.pinyin_tokens.t2id[i])
                for i in txt:
                    y_.append(self.hans_tokens.t2id[i])
                x_.append(self.pinyin_tokens.endid())
                y_.append(self.hans_tokens.endid())
                x.append(np.array(x_))
                y.append(np.array(y_))
            except:
                continue
        return  x,y
    def bert_decode(self,x,x2=None):
        tokens,segs=[],[]
        if x2 is not None:
            for i,j in zip(x,x2):
                t,s=self.bert_token.encode(''.join(i))
                index=np.where(j==2)[0]
                if len(index)>0:
                    for n in index:
                        t[int(n)]=103
                tokens.append(t)
                segs.append(s)
        else:
            for i in x:
                t,s=self.bert_token.encode(''.join(i))
                tokens.append(t)
                segs.append(s)
        return tokens,segs

    def pad(self,x,mode=1):
        length=0

        for i in x:
            length=max(length,len(i))
        if mode==2:
            for i in range(len(x)):
                pading=np.ones([length-len(x[i]),x[i].shape[1]])*-10.
                x[i]=np.vstack((x[i],pading))

        else:
            x=pad_sequences(x,length,padding='post',truncating='post')
        return x
    def get_bert_feature(self,bert_t,bert_s):
        f=[]
        for t,s in zip(bert_t,bert_s):
            t=np.expand_dims(np.array(t),0)
            s=np.expand_dims(np.array(s),0)
            feature=self.bert.predict([t,s])
            f.append(feature[0])
        return f

    def generate(self,batch):
        sample=np.random.randint(0,len(self.texts),batch)
        trainx=[pypinyin.pinyin(self.texts[i]) for i in sample]
        trainy=[self.texts[i] for i in sample]
        x,y=self.preprocess(trainx,trainy)
        e_bert_t, e_bert_s = self.bert_decode(trainy)
        e_features=self.get_bert_feature(e_bert_t, e_bert_s)
        x=self.pad(x)
        y=self.pad(y)
        e_features=self.pad(e_features,2)


        x=np.array(x)
        y=np.array(y)
        e_features=np.array(e_features,dtype='float32')

        return x,y,e_features[:,1:]