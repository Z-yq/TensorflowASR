
import tensorflow as tf

import collections
from AMmodel.wav_model import WavePickModel
from AMmodel.stream_transducer_wrap import Transducer
from AMmodel.stream_ctc_wrap import CtcModel
from utils.tools import merge_two_last_dims,split_two_first_dims,merge_two_first_dims
from AMmodel.layers.multihead_attention import MultiHeadAttention

class GLU(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 name="glu_activation",
                 **kwargs):
        super(GLU, self).__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        a, b = tf.split(inputs, 2, axis=self.axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)

    def get_config(self):
        conf = super(GLU, self).get_config()
        conf.update({"axis": self.axis})
        return conf

class ConvSubsampling(tf.keras.layers.Layer):
    def __init__(self,
                 odim: int,
                 reduction_factor: int = 4,
                 dropout: float = 0.0,
                 name="conv_subsampling",
                 **kwargs):
        super(ConvSubsampling, self).__init__(name=name, **kwargs)
        assert reduction_factor % 2 == 0, "reduction_factor must be divisible by 2"
        self.conv1 = tf.keras.layers.Conv2D(
            filters=odim, kernel_size=(3, 3),
            strides=((reduction_factor // 2), 2),
            padding="same", activation="relu"
        )
        # self.conv1=Involution2D( filters=odim, kernel_size= 3,
        #     strides=((reduction_factor // 2), 2),
        #     padding="same", )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=odim, kernel_size=(3, 3),
            strides=(2, 2), padding="same",
            activation="relu"
        )
        self.linear = tf.keras.layers.Dense(odim)
        self.do = tf.keras.layers.Dropout(dropout)

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv1(inputs, training=training)
        outputs = self.conv2(outputs, training=training)

        outputs = merge_two_last_dims(outputs)
        outputs = self.linear(outputs, training=training)
        return self.do(outputs, training=training)

    def get_config(self):
        conf = super(ConvSubsampling, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.conv2.get_config())
        conf.update(self.linear.get_config())
        conf.update(self.do.get_config())
        return conf


class FFModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 name="ff_module",
                 **kwargs):
        super(FFModule, self).__init__(name=name, **kwargs)
        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization()
        self.ffn1 = tf.keras.layers.Dense(4 * input_dim)
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name="swish_activation")
        self.do1 = tf.keras.layers.Dropout(dropout)
        self.ffn2 = tf.keras.layers.Dense(input_dim)
        self.do2 = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([inputs, self.fc_factor * outputs])
        return outputs

    def get_config(self):
        conf = super(FFModule, self).get_config()
        conf.update({"fc_factor": self.fc_factor})
        conf.update(self.ln.get_config())
        conf.update(self.ffn1.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.do2.get_config())
        conf.update(self.res_add.get_config())
        return conf


class MHSAModule(tf.keras.layers.Layer):
    def __init__(self,
                 dmodel,
                 head_size,
                 num_heads,
                 dropout=0.0,
                 name="mhsa_module",
                 **kwargs):
        super(MHSAModule, self).__init__(name=name, **kwargs)
        # self.pc = PositionalEncoding()
        self.ln = tf.keras.layers.LayerNormalization()
        self.mha = MultiHeadAttention(head_size=head_size, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(head_size=head_size, num_heads=num_heads)
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs,key, training=False, **kwargs):
        # outputs = self.pc(inputs)
        outputs = self.ln(inputs, training=training)
        outputs = self.mha2([outputs,outputs,outputs], training=training)
   
        outputs = self.mha([outputs,key,key], training=training)
        # print(outputs.shape)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(MHSAModule, self).get_config()
        # conf.update(self.pc.get_config())
        conf.update(self.ln.get_config())
        conf.update(self.mha.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf


class ConvModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 kernel_size=32,
                 dropout=0.0,
                 name="conv_module",
                 **kwargs):
        super(ConvModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization()
        self.pw_conv_1 = tf.keras.layers.Conv1D(
            filters=2 * input_dim, kernel_size=1, strides=1,
            padding="same", name="pw_conv_1"
        )

        self.glu = GLU()
        self.dw_conv = tf.keras.layers.SeparableConv1D(
            filters=2 * input_dim, kernel_size=kernel_size, strides=1,
            padding="same", depth_multiplier=1, name="dw_conv"
        )
        self.bn =tf.keras.layers.BatchNormalization()
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name="swish_activation")

        self.pw_conv_2 = tf.keras.layers.Conv1D(filters=input_dim, kernel_size=1, strides=1,
                                                padding="same", name="pw_conv_2")
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(ConvModule, self).get_config()
        conf.update(self.ln.get_config())
        conf.update(self.pw_conv_1.get_config())
        conf.update(self.glu.get_config())
        conf.update(self.dw_conv.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.pw_conv_2.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 head_size=144,
                 num_heads=4,
                 kernel_size=32,
                 name="conformer_block",
                 **kwargs):
        super(ConformerBlock, self).__init__(name=name, **kwargs)
        self.ffm1 = FFModule(input_dim=input_dim,
                             dropout=dropout, fc_factor=fc_factor,
                             name="ff_module_1")
        self.mhsam = MHSAModule(input_dim,head_size=head_size, num_heads=num_heads,
                                dropout=dropout)
        self.convm = ConvModule(input_dim=input_dim, kernel_size=kernel_size,
                                dropout=dropout)
        self.ffm2 = FFModule(input_dim=input_dim,
                             dropout=dropout, fc_factor=fc_factor,
                             name="ff_module_2")
        self.ln = tf.keras.layers.LayerNormalization()

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, key,training=False, **kwargs):
        outputs = self.ffm1(inputs, training=training)
        outputs = self.mhsam(outputs,key=key, training=training)
        outputs = self.convm(outputs, training=training)
        outputs = self.ffm2(outputs, training=training)
        outputs = self.ln(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(ConformerBlock, self).get_config()
        conf.update(self.ffm1.get_config())
        conf.update(self.mhsam.get_config())
        conf.update(self.convm.get_config())
        conf.update(self.ffm2.get_config())
        conf.update(self.ln.get_config())
        return conf


class StreamingEncoderCell(tf.keras.layers.AbstractRNNCell):
    """Streaming custom Encoder cell."""

    def __init__(self, dmodel=144,
                 num_blocks=16,
                 head_size=36,
                 num_heads=4,
                 kernel_size=32,
                 fc_factor=0.5,
                 dropout=0.0,
                 name="conformer_encoder",
                 **kwargs):
        """Init variables."""
        super().__init__(**kwargs)

        self.encoder=ConformerEncoder(
            dmodel=dmodel,
            num_blocks=num_blocks,
            head_size=head_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            fc_factor=fc_factor,
            dropout=dropout,
            name=name,
        )
        self.lstm_cell=tf.keras.layers.LSTMCell(dmodel,dropout=dropout)
        self.gru_cell=tf.keras.layers.GRUCell(dmodel,dropout=dropout)
        self.dmodel=dmodel
    @property
    def state_size(self):
        """Return hidden state size."""
        return [tf.TensorShape([self.dmodel]),tf.TensorShape([self.dmodel]),tf.TensorShape([self.dmodel])]
    @property
    def output_size(self):
        return tf.TensorShape([None,self.dmodel])

    def get_initial_state(self, batch_size,inputs=None,dtype=None,):
        """Get initial states."""

        initial_key = tf.zeros(shape=[batch_size, self.dmodel ], dtype=tf.float32)

        initial_state_lh,initial_state_lc = tf.zeros([batch_size,self.dmodel],dtype=tf.float32), tf.zeros([batch_size,self.dmodel],dtype=tf.float32)

        return [initial_state_lh,initial_state_lc,initial_key]


    def call(self, inputs, states):
        """Call logic."""

        decoder_input = inputs

        key = states[-1]

        B=tf.shape(decoder_input)[0]



        key_reshape=tf.reshape(key,[B,-1,self.dmodel])

        encoder_out=self.encoder([decoder_input,key_reshape])
        T=tf.shape(encoder_out)[1]
        lh=states[0]
        lc=states[1]
        gh=tf.zeros_like(lh)
        for i in range(T):
            _,gh=self.gru_cell(encoder_out[:,i],(gh))

        _,(lh,lc)=self.lstm_cell(gh,(lh,lc))

        new_states = [lh,lc,lc]

        return encoder_out, new_states

class ConformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 dmodel=144,
                 num_blocks=16,
                 head_size=36,
                 num_heads=4,
                 kernel_size=32,
                 fc_factor=0.5,
                 dropout=0.0,
                 name="conformer_encoder",
                 **kwargs):
        super(ConformerEncoder, self).__init__(name=name+'blocks', **kwargs)
        self.conformer_blocks=[]
        for i in range(num_blocks):
            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                name=f"conformer_block_{i}"
            )
            self.conformer_blocks.append(conformer_block)

    # @tf.function()
    def call(self, inputs, training=False, **kwargs):
        # input with shape [B, T, V1, V2]
        outputs,key=inputs

        for cblock in self.conformer_blocks:
            outputs = cblock(outputs,key, training=training)

        return outputs

    def get_config(self):
        conf = super(ConformerEncoder, self).get_config()

        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf
class StreamingConformerEncoder(tf.keras.Model):


    def __init__(self,
                 dmodel=144,
                 reduction_factor=4,
                 num_blocks=4,
                 cell_nums=4,
                 head_size=36,
                 num_heads=4,
                 kernel_size=32,
                 fc_factor=0.5,
                 dropout=0.0,
                 add_wav_info=False,
                 hop_size=80,
                 name="streaming_conformer_encoder",
                 **kwargs):
        """Initial variables."""
        super(StreamingConformerEncoder, self).__init__()
        self.dmodel = dmodel
        self.reduction_factor = reduction_factor
        self.conv_subsampling = ConvSubsampling(
            odim=dmodel, reduction_factor=reduction_factor,
            dropout=dropout
        )
        self.dropout = dropout
        self.cell_nums=cell_nums
        self.add_wav_info = add_wav_info
        if self.add_wav_info:
            self.wav_layer = WavePickModel(dmodel, hop_size)
        cells=[]
        for i in range(cell_nums):
            cells.append(StreamingEncoderCell(dmodel=dmodel,

            num_blocks=num_blocks,
            head_size=head_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            fc_factor=fc_factor,
            dropout=dropout,
            name=name+'cell_%s'%i,))

        self.custom_layer = tf.keras.layers.RNN(cells, return_sequences=True, return_state=True,
                                                 name='customer_rnn')

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs,states=None, training=None, mask=None):

        if self.add_wav_info:
            mel_inputs, wav_inputs = inputs
            B=tf.shape(mel_inputs)[0]
            T=tf.shape(mel_inputs)[1]
            mel_inputs=merge_two_first_dims(mel_inputs)
            wav_inputs=merge_two_first_dims(wav_inputs)
            mel_outputs = self.conv_subsampling(mel_inputs, training=training)
            wav_outputs = self.wav_layer(wav_inputs, training=training)
            outputs = mel_outputs+wav_outputs
            outputs=split_two_first_dims(outputs,B,T)
        else:
            mel_inputs=inputs
            B = tf.shape(mel_inputs)[0]
            T = tf.shape(mel_inputs)[1]
            mel_inputs=merge_two_first_dims(mel_inputs)
            outputs = self.conv_subsampling(mel_inputs, training=training)
            outputs = split_two_first_dims(outputs, B, T)

        if states is None:

            states=self.custom_layer.get_initial_state(outputs)

        outputs=self.custom_layer(outputs,initial_state=states)
        return outputs[0],outputs[1:]
    def get_init_states(self,inputs):
        return self.custom_layer.get_initial_state(inputs)
    def inference(self,inputs,states):
        if self.add_wav_info:
            mel_inputs, wav_inputs = inputs

            mel_outputs = self.conv_subsampling(mel_inputs, training=False)
            wav_outputs = self.wav_layer(wav_inputs, training=False)
            outputs = mel_outputs+wav_outputs

        else:
            mel_inputs=inputs

            outputs = self.conv_subsampling(mel_inputs, training=False)
        outputs=tf.expand_dims(outputs,1)
        outputs=self.custom_layer(outputs,initial_state=states)
        new_states=outputs[1:]
        result=tf.squeeze(outputs[0],1)
        return result,new_states
    def get_config(self):
        conf = super(StreamingConformerEncoder, self).get_config()
        conf.update(self.conv_subsampling.get_config())
        if self.add_wav_info:
            conf.update(self.wav_layer.get_config())
        conf.update(self.custom_layer.get_config())


class StreamingConformerTransducer(Transducer):
    def __init__(self,
                 dmodel: int,
                 reduction_factor: int,
                 vocabulary_size: int,
                 num_blocks: int = 8,
                 cell_nums: int =4,
                 head_size: int = 512,
                 num_heads: int = 4,
                 kernel_size: int = 32,
                 fc_factor: float = 0.5,
                 dropout: float = 0,
                 embed_dim: int = 512,
                 embed_dropout: int = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 512,
                 joint_dim: int = 1024,
                 name: str = "conformer_transducer",
                 speech_config=dict,
                 **kwargs):
        super(StreamingConformerTransducer, self).__init__(
            encoder=StreamingConformerEncoder(
                dmodel=dmodel,
                reduction_factor=reduction_factor,
                num_blocks=num_blocks,
                cell_nums=cell_nums,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                fc_factor=fc_factor,
                dropout=dropout,
                add_wav_info=speech_config['add_wav_info'],
                hop_size=int(speech_config['stride_ms'] * speech_config['sample_rate'] // 1000)*reduction_factor,
            ),
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_lstms=num_lstms,
            lstm_units=lstm_units,
            joint_dim=joint_dim,
            name=name, speech_config= speech_config, **kwargs
        )
        self.time_reduction_factor = reduction_factor

class StreamingConformerCTC(CtcModel):
    def __init__(self,
                 dmodel: int,
                 reduction_factor: int,
                 vocabulary_size: int,
                 cell_nums:int =4,
                 num_blocks: int = 16,
                 head_size: int = 36,
                 num_heads: int = 4,
                 kernel_size: int = 32,
                 fc_factor: float = 0.5,
                 dropout: float = 0,
                 name='conformerCTC',
                 speech_config=dict,
                 **kwargs):
        super(StreamingConformerCTC, self).__init__(
            encoder=StreamingConformerEncoder(
                dmodel=dmodel,
                reduction_factor=reduction_factor,
                cell_nums=cell_nums,
                num_blocks=num_blocks,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                fc_factor=fc_factor,

                dropout=dropout,
                add_wav_info=speech_config['add_wav_info'],
                hop_size=int(speech_config['stride_ms'] * speech_config['sample_rate']//1000)*reduction_factor,
            ),num_classes=vocabulary_size,name=name,speech_config=speech_config)
        self.time_reduction_factor = reduction_factor

if __name__ == '__main__':

    import time
    model=StreamingConformerEncoder()
    x=tf.random.uniform([3,4,50,80,1])
    x2=tf.random.uniform([3,4,4000,1])
    # model.setup_init_state(3)
    out1=model(x,training=True)
    states=model.custom_layer.get_initial_state(x)
    for i in range(x.shape[1]):
        s=time.time()
        out1,states=model.inference(x[:,i],states)
        e=time.time()
        print('cost time',e-s,out1.shape)
    # out2=model(x,training=True)
    model.summary()
    # print(out2[1])
