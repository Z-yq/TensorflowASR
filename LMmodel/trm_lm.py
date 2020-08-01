from LMmodel.tf2_trm import Transformer,create_masks
import tensorflow as tf
import os
import logging
import numpy as np
from utils.token_tool import MakeS2SDict
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
class LM():
    def __init__(self,hparams):
        self.hp=hparams
        self.pinyin_tokens, self.hans_tokens = MakeS2SDict(None, delimiter=' ', dict_file=self.hp.lm_dict_file,model='lm')
        self.model = Transformer(
            num_layers=self.hp.lm_layers, d_model=self.hp.lm_d_model, num_heads=self.hp.lm_heads, dff=self.hp.lm_dff,
            input_vocab_size=self.pinyin_tokens.num(), target_vocab_size=self.hans_tokens.num(),
            pe_input=self.hp.lm_en_max, pe_target=self.hp.lm_de_max, rate=self.hp.lm_dp)
        try:
            self.model.load_weights(os.path.join(self.hp.lm_save_path,'asr_lm'))
        except:
            logging.info('lm loading model failed.')
    def encode(self,word,token):
        x=[2]
        for i in word:
            x.append(token.t2id[i])
        x.append(3)
        return np.array(x)[np.newaxis,:]
    def decode(self,out,token):
        de=[]
        for i in out[1:]:
            de.append(token.id2t[i])
        return de
    def batch_decode(self,outs,token):
        de=[]
        for out in outs:
            de_=[]
            for i in out[1:]:
                if i==3:
                    break
                else:
                    de_.append(token.id2t[i])
            de.append(de_)
        return de
    def batch_encode(self,words,token):

        x=[]
        maxlen=0
        for word in words:
            x_ = [2]
            for i in word:
                x_.append(token.t2id[i[0] if isinstance(i,list) else i])
            x_.append(3)
            maxlen=max(len(x_),maxlen)

            x.append(np.array(x_))
        x=tf.keras.preprocessing.sequence.pad_sequences(x,maxlen,padding='post',truncating='post')
        return x

    def batch_eval(self,encoder_input,transformer):
        batch=encoder_input.shape[0]
        output=np.ones([batch,1],dtype='int32')*2
        maxlen=encoder_input.shape[1]
        for i in range(maxlen):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, _= transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # 从 seq_len 维度选择最后一个词
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # 如果 predicted_id 等于结束标记，就返回结果
            # if np.mean(predicted_id==3)==1.:
            #     # print('id out')
            #     return output

            # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
            output = tf.concat([output, predicted_id], axis=-1)


        return output
    def evaluate(self,encoder_input,transformer):



        # 输入语句是葡萄牙语，增加开始和结束标记



        # 因为目标是英语，输入 transformer 的第一个词应该是
        # 英语的开始标记。

        decoder_input = [2]
        output = tf.expand_dims(decoder_input, 0)
        maxlen=encoder_input.shape[1]+10
        for i in range(maxlen):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, _= transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # 从 seq_len 维度选择最后一个词
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # 如果 predicted_id 等于结束标记，就返回结果
            if predicted_id == 3:
                # print('id out')
                return tf.squeeze(output, axis=0)

            # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
            output = tf.concat([output, predicted_id], axis=-1)


        return tf.squeeze(output, axis=0)
    def get(self,pins):
        x=self.encode(pins,self.pinyin_tokens)
        out=self.evaluate(x,self.model)
        result=self.decode(out,self.hans_tokens)
        return result



