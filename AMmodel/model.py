import tensorflow as tf
from utils.token_tool import ITokens,MakeS2SDict
from utils import audio
from AMmodel.switchnorm import SwitchNormalization
import numpy as np
import os

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_decode_func(args):
    y_pred, input_length = args

    (out, _) = tf.keras.backend.ctc_decode(y_pred, tf.keras.backend.squeeze(input_length, axis=-1), greedy=True,
                                           beam_width=5)
    return out

def resnet(dim):
    x = [tf.keras.layers.Conv1D(dim * 2, 3, padding='same', activation='elu' ),
         tf.keras.layers.Conv1D(dim * 2, 3, padding='same', activation='elu'),
         tf.keras.layers.Conv1D(dim, 3, padding='same', activation='elu'),
         SwitchNormalization(),tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dim,activation='elu'))]
    x2 = [tf.keras.layers.Conv1D(int(dim / 2), 3, padding='same', activation='elu'),
          tf.keras.layers.Conv1D(int(dim / 2), 7, padding='same', activation='elu'),
          tf.keras.layers.Conv1D(dim, 3, padding='same', activation='elu'),
          SwitchNormalization(),tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dim,activation='elu'))]
    x3 = tf.keras.layers.Conv1D(dim, 5, activation='elu', padding='same')
    return x, x2, x3

class Encoder_Block(tf.keras.layers.Layer):
    def __init__(self, output_dim, dropout, layer, train, name, last=False):
        super(Encoder_Block, self).__init__()
        self.block_units = output_dim
        self.dropout = dropout
        self.layer = layer
        self.train = train
        self.last = last
        self.SepConv1D = tf.keras.layers.SeparableConv1D(self.block_units, 5, padding='same', activation='elu',
                                                         name=name + 'Separable')
        self.sw1 = SwitchNormalization()
        self.conv_block = [
            tf.keras.layers.Conv1D(self.block_units, 5, padding='same', name=name + 'Conv%d' % i, activation='elu') for
            i in range(1, self.layer + 1)]

        self.TRM_Encoder = [tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.block_units,activation='elu')) for _ in
                                                range(1, self.layer + 1)]
        self.cnn1, self.cnn2, self.cnn3 = resnet(self.block_units)

        self.sws=[SwitchNormalization() for _ in range(1, self.layer + 1)]
        self.pooling2 = tf.keras.layers.Conv1D(self.block_units, 4, 2, 'same', use_bias=False)


    def call(self, inputs,training=False):
        x = self.SepConv1D(inputs)
        x = self.sw1(x,training=training)

        for i in range(self.layer):
            plus=x
            x = self.conv_block[i](x)
            x = self.TRM_Encoder[i](x)
            x+=plus
            x=self.sws[i](x,training=training)

        if self.last:
            pass
        else:
            x = self.pooling2(x)
        out1 = x
        out2 = x
        for i in self.cnn1:
            out1 = i(out1,training=training)
        for j in self.cnn2:
            out2 = j(out2,training=training)
        x = out1 + out2 + x
        x = self.cnn3(x)
        return x


class Voice_Block(tf.keras.layers.Layer):
    def __init__(self, output_dim, units):
        super(Voice_Block, self).__init__()

        self.value = tf.keras.layers.LSTM(units, return_sequences=True, name='value')
        self.value_dense = tf.keras.layers.Dense(output_dim, activation='elu', name='value_dense')
        self.picth = tf.keras.layers.LSTM(units, return_sequences=True, name='pitch')
        self.picth_dense = tf.keras.layers.Dense(output_dim, activation='elu', name='picth_dense')
        self.voice = tf.keras.layers.LSTM(units, return_sequences=True, name='voice')
        self.voice_dense = tf.keras.layers.Dense(output_dim, activation='elu', name='voice_dense')

    def call(self, inputs,training=None):
        f1 = self.value(inputs)
        f2 = self.picth(inputs)
        f3 = self.voice(inputs)
        f1 = f1 - f2 - f3
        f2 = f2 - f1 - f3
        f3 = f3 - f1 - f2
        out1 = self.value_dense(f1)
        out2 = self.picth_dense(f2)
        out3 = self.voice_dense(f3)
        out = out1 + out2 + out3
        out = tf.nn.tanh(out)
        return out
class STTmodel(tf.keras.Model):
    def __init__(self, input_dim=128, ctc_output_dim=[48, 48, 48], rnn_uints=[768,768, 768], layers=1, dp=0.1,
                 train=True, wav_encoder_num=512):
        super(STTmodel, self).__init__()
        self.input_dim = input_dim
        self.ctc_output_dim = ctc_output_dim
        self.block_units = rnn_uints
        self.layer = layers
        self.dropout = dp
        self.train = train
        self.wav_encoder = tf.keras.Sequential([tf.keras.layers.Conv1D(128, 4, strides=2, padding='same'),
                                                tf.keras.layers.Conv1D(128, 4, strides=2, padding='same'),
                                                tf.keras.layers.Conv1D(128, 4, strides=2, padding='same'),
                                                tf.keras.layers.Conv1D(128, 4, strides=2, padding='same'),
                                                tf.keras.layers.Conv1D(128, 10, strides=5, padding='same',
                                                                       activation='elu'),
                                                tf.keras.layers.Conv1D(wav_encoder_num, 3, padding='causal',
                                                                       activation='elu'),
                                                tf.keras.layers.Conv1D(wav_encoder_num, 5, padding='causal',
                                                                       activation='elu'),
                                                ], name='wav_encoder')
        self.wav_c1, self.wav_c2, self.wav_c3 = resnet(wav_encoder_num)
        self.wav_decoder = tf.keras.layers.Conv1D(input_dim, 3, padding='causal', activation='tanh')
        self.block1 = Encoder_Block(self.block_units[0], self.dropout, self.layer, self.train, name='ctc1')
        self.block2 = Encoder_Block(self.block_units[1], self.dropout, self.layer, self.train, name='ctc2')
        self.block3 = Encoder_Block(self.block_units[2], self.dropout, self.layer, self.train, name='ctc3', last=True)
        self.block4 = tf.keras.Sequential(
            [tf.keras.layers.Conv1D(self.block_units[1], 3, padding='same', activation='elu'),
            tf.keras.layers.Conv1D(self.block_units[1],4,2,padding='same',use_bias=False),
             tf.keras.layers.Conv1D(self.block_units[1], 3, padding='same',
                                    activation='elu'),
             ])
        self.prenet = tf.keras.layers.Dense(self.input_dim, activation='tanh', name='prenet')
        self.dense1 = tf.keras.layers.Dense(self.input_dim, activation='elu', name='Flatten%d' % 1)
        self.dense2 = tf.keras.layers.Dense(self.input_dim, activation='elu', name='Flatten%d' % 2)
        self.voice_vector = Voice_Block(input_dim, 256)

        self.ctc_out1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.ctc_output_dim[0], activation='softmax', name='ctc%d_out' % 1))
        self.ctc_out2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.ctc_output_dim[1], activation='softmax', name='ctc%d_out' % 2))
        self.ctc_out3 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.ctc_output_dim[2], activation='softmax', name='ctc%d_out' % 3))


    @tf.function(experimental_relax_shapes=True)
    def call(self, input_data, training=None, mask=None):

        inputs, wavs= input_data
        x = self.wav_encoder(wavs)
        voice = self.voice_vector(x)

        out1 = x
        out2 = x
        for i in self.wav_c1:
            out1 = i(out1,training=training)
        for j in self.wav_c2:
            out2 = j(out2,training=training)
        wav_f = out1 + out2 + x
        wav_f = self.wav_c3(wav_f)
        wav_f = self.wav_decoder(wav_f)
        inputs = self.prenet(inputs)

        x = wav_f+voice+inputs

        b1 = self.block1(x,training=training)

        ctc1_out = self.ctc_out1(b1)

        b2 = self.block2(x,training=training)

        ctc2_out = self.ctc_out2(b2)


        org = self.block4(x)

        add =tf.concat([org,b1,b2],-1)
        b3 = self.block3(add,training=training)
        ctc3_out = self.ctc_out3(b3)
        return ctc1_out,ctc2_out,ctc3_out





class AM():
    def __init__(self,hparams):
        self.hp= hparams
        self.itokens3, _ = MakeS2SDict(None, dict_file=self.hp.am_dict_file, model='stt')
        _, self.itokens4 = MakeS2SDict(None, dict_file=self.hp.am_dict_file, model='correct')
        self.itokens1 = ITokens(list('qwertyuiopasdfghjklzxcvbnm1234'))
        self.itokens2 = ITokens(
            ['x', 'j', 'ie1', 'i1', 'ei1', 'van1', 'vn1', 'ou1', 'uai4', 'uei2', 'v2', 'ao4', 'iao4', 'in2', 'a4',
             'eng',
             'l', 'van4', 've3', 'ian2', 'en1', 'e', 'i3', 'vn4', 'ei3', 'in3', 'o', 'ou', 'ai', 'ang1', 'ang2', 'uo',
             'h',
             'in4', 'an4', 'iao2', 'uo2', 'ai1', 'eng2', 'van3', 'zh', 'a2', 'ing4', 'uen3', 'iang1', 'ie2', 'ao',
             'ua3',
             'ou4', 'uen2', 'eng4', 'uai1', 'ai2', 's', 'iong2', 'ua1', 'iang2', 'v1', 'i', 'iou3', 'uang2', 'iong3',
             'en2',
             'ueng1', 'ie3', 'ing2', 'en', 'an2', 'iao3', 'uo1', 'q', 'a3', 'er2', 'er3', 'ing1', 'uai2', 'ueng4',
             'iong4',
             'uei4', 'o1', 'r', 'uen4', 'c', 'an', 'ian4', 'ei2', 'vn2', 'ing3', 't', 'm', 'ang4', 'uan3', 'ia2', 'er',
             'ia4', 'ei4', 'en4', 'ai4', 'ong2', 'u2', 'ua4', 'i2', 'i4', 'u', 'o4', 'an1', 'er4', 'f', 'io1', 'ia1',
             'v3',
             'uen1', 'uai3', 'ia', 'iou1', 'e4', 'uang1', 'iao1', 'iong1', 'uan2', 'ie', 'eng1', 'ie4', 'uei', 'ao1',
             'in1',
             'vn3', 'u3', 'uan1', 'd', 'uei3', 'ao2', 've1', 'iou4', 'ang', 'a', 'uei1', 'v', 'z', 'ua', 'ian3', 'n',
             'ou2',
             'ou3', 'in', 'iang3', 'ang3', 'en3', 'uang4', 'ong4', 'v4', 'ei', 'u1', 'g', 'iang4', 'u4', 'e2', 'o3',
             'ia3',
             'ueng3', 'ong3', 'k', 'ong1', 'an3', 'ua2', 'a1', 'ai3', 'uo4', 'ian1', 'uang3', 've2', 'b', 'p', 'eng3',
             'van2', 've4', 'sh', 'ao3', 'uo3', 'ch', 'uen', 'uan4', 'o2', 'n2', 'e3', 'iang', 'e1', 'iou2']

        )
        self.STT = STTmodel(input_dim=self.hp.num_mels,
                            ctc_output_dim=[self.itokens1.num() + 1, self.itokens2.num() + 1, self.itokens3.num() + 1],
                            rnn_uints=self.hp.am_block_units, dp=self.hp.am_dp, layers=self.hp.am_layers)
        try:
            self.STT.load_weights(os.path.join(self.hp.am_save_path, 'stt'))
        except:
            print('am resume failed.')
            pass
    def decode_result(self,result,itoken):
        de=[]

        for i in result[0]:
            # print(result,i)
            de.append(itoken.id2t[int(i)])
        return de


    def model_infrence(self,x):

        ctc1, ctc2, ctc3 = self.STT(x[:2],training=False)
        result = ctc_decode_func([ctc3, x[-1]])


        return result
    def predict(self,fp):
        if '.pcm' in fp:
            data=np.fromfile(fp,'int16')
            data=np.array(data,'float32')
            data/=32768
        else:
            data = audio.load_wav(fp, sr=self.hp.sample_rate)

        data = audio.preemphasis(data, 0.97)
        data /= np.abs(data).max()
        mel = audio.melspectrogram(data, self.hp)
        mel = mel.transpose((1, 0))

        if len(data) % self.hp.hop_size == 0:
            mel = mel[:-1]
        x = np.expand_dims(mel, 0)
        data = data.reshape([1, -1, 1])

        result = self.model_infrence([x.astype('float32'), data.astype('float32'), np.array([[x.shape[1] // 2]]).astype('int32')])
        result = self.decode_result(result[0].numpy(), self.itokens3)
        return result