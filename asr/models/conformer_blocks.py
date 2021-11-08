
import tensorflow as tf
from asr.models.wav_model import WavePickModel
from utils.tools import merge_two_last_dims
from asr.models.layers.switchnorm import SwitchNormalization
from asr.models.layers.multihead_attention import MultiHeadAttention
from asr.models.layers.time_frequency import Spectrogram,Melspectrogram
from asr.models.layers.positional_encoding import PositionalEncoding
from leaf_audio import frontend
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
                 head_size,
                 num_heads,
                 dropout=0.0,
                 name="mhsa_module",
                 **kwargs):
        super(MHSAModule, self).__init__(name=name, **kwargs)
        # self.pc = PositionalEncoding()
        self.ln = tf.keras.layers.LayerNormalization()
        self.mha = MultiHeadAttention(head_size=head_size, num_heads=num_heads)
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        # outputs = self.pc(inputs)
        outputs = self.ln(inputs, training=training)
        outputs = self.mha([outputs, outputs, outputs], training=training)
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
        self.bn =SwitchNormalization()
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
        self.mhsam = MHSAModule(head_size=head_size, num_heads=num_heads,
                                dropout=dropout)
        self.convm = ConvModule(input_dim=input_dim, kernel_size=kernel_size,
                                dropout=dropout)
        self.ffm2 = FFModule(input_dim=input_dim,
                             dropout=dropout, fc_factor=fc_factor,
                             name="ff_module_2")
        self.ln = tf.keras.layers.LayerNormalization()

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        outputs = self.ffm1(inputs, training=training)
        outputs = self.mhsam(outputs, training=training)
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


class ConformerEncoder(tf.keras.Model):
    def __init__(self,
                 dmodel=144,
                 reduction_factor=4,
                 num_blocks=16,
                 head_size=36,
                 num_heads=4,
                 kernel_size=32,
                 fc_factor=0.5,
                 dropout=0.0,
                 add_wav_info=False,
                 sample_rate=16000,
                 n_mels=80,
                 mel_layer_type='leaf',
                 mel_layer_trainable=False,
                 stride_ms=10,
                 name="conformer_encoder",
                 **kwargs):
        super(ConformerEncoder, self).__init__(name=name, **kwargs)
        self.dmodel=dmodel
        self.num_heads=num_heads
        self.fc_factor=fc_factor
        self.dropout=dropout
        self.head_size=head_size
        self.hop_size=int(stride_ms * sample_rate // 1000)*reduction_factor
        self.add_wav_info = add_wav_info
        self.reduction_factor=reduction_factor
        self.conv_subsampling = ConvSubsampling(
            odim=dmodel, reduction_factor=reduction_factor,
            dropout=dropout
        )

        if mel_layer_type == 'Melspectrogram':
            self.mel_layer = Melspectrogram(sr=sample_rate, n_mels=n_mels,
                                            n_hop=int(stride_ms * sample_rate // 1000),
                                            n_dft=1024,
                                            trainable_fb=mel_layer_trainable
                                            )
        elif mel_layer_type=='leaf':
            self.mel_layer=frontend.Leaf(n_filters=n_mels,sample_rate=sample_rate,window_stride=stride_ms,complex_conv_init=frontend.initializers.GaborInit(sample_rate=sample_rate,min_freq=30*(sample_rate//800),
                                                                                                                                           max_freq=3900*(sample_rate//8000)))
        else:
            self.mel_layer = Spectrogram(
                n_hop=int(stride_ms* sample_rate// 1000),
                n_dft=1024,
                trainable_kernel=mel_layer_trainable
            )
        self.mel_layer.trainable =mel_layer_trainable
        if self.add_wav_info:
            self.wav_layer=WavePickModel(dmodel,self.hop_size)
        self.conformer_blocks = []
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
    def _build(self):
        fake=tf.random.uniform([1,16000,1])
        self(fake)

    def call(self, inputs, training=False, **kwargs):
        if self.add_wav_info:
            mel_inputs=self.mel_layer(inputs)
            mel_outputs = self.conv_subsampling(mel_inputs, training=training)
            wav_outputs = self.wav_layer(inputs, training=training)
            outputs = mel_outputs+wav_outputs
        else:
            inputs=self.mel_layer(inputs)
            outputs = self.conv_subsampling(inputs, training=training)

        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, training=training)

        return outputs

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[
                     tf.TensorSpec([None, None, 1], dtype=tf.int32),
                 ]
                 )
    def inference(self,inputs):
        if self.add_wav_info:
            mel_inputs=self.mel_layer(inputs)
            mel_outputs = self.conv_subsampling(mel_inputs, training=False)
            wav_outputs = self.wav_layer(inputs, training=False)
            outputs = mel_outputs+wav_outputs
        else:
            inputs=self.mel_layer(inputs)
            outputs = self.conv_subsampling(inputs, training=False)

        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, training=False)

        return outputs
    def get_config(self):
        conf = super(ConformerEncoder, self).get_config()
        conf.update(self.conv_subsampling.get_config())
        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf
class CTCDecoder(tf.keras.Model):
    def __init__(self,num_classes,
                 dmodel=144,
                 num_blocks=16,
                 head_size=36,
                 num_heads=4,
                 fc_factor=0.5,
                 dropout=0.0,
                 kernel_size=32,
                 **kwargs
                 ):
        super(CTCDecoder, self).__init__()
        self.decode_layers = []
        self.dmodel=dmodel
        self.project=tf.keras.layers.Dense(dmodel)
        for i in range(num_blocks):
            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                name=f"decoder_conformer_block_{i}"
            )
            self.decode_layers.append(conformer_block)

        self.fc = tf.keras.layers.Dense(units=num_classes, activation="linear",
                                  use_bias=True, name="fully_connected")

    def _build(self):
        fake=tf.random.uniform([1,10,self.dmodel])
        self(fake)

    def call(self, inputs, training=None, mask=None):
        outputs=self.project(inputs,training=training)
        for layer in self.decode_layers:
            outputs = layer(outputs, training=training)
        outputs = self.fc(outputs, training=training)
        return outputs

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[
                     tf.TensorSpec([None, None,144], dtype=tf.int32),
                 ]
                 )
    def inference(self,inputs):
        outputs = self.project(inputs, training=False)
        for layer in self.decode_layers:
            outputs = layer(outputs, training=False)
        outputs = self.fc(outputs, training=False)
        return outputs
class RMHSAModule(tf.keras.layers.Layer):
    def __init__(self,
                 head_size,
                 num_heads,
                 dropout=0.0,
                 name="mhsa_module",
                 **kwargs):
        super(RMHSAModule, self).__init__(name=name, **kwargs)
        self.pc = PositionalEncoding()
        self.ln = tf.keras.layers.LayerNormalization()
        self.mha = MultiHeadAttention(head_size=head_size, num_heads=num_heads)
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()


    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs,enc, training=False, **kwargs):
        outputs = self.pc(inputs)
        outputs = self.ln(outputs, training=training)
        # print(outputs.shape)
        outputs = self.mha([outputs, enc, enc], training=training)
        # print(outputs.shape)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(RMHSAModule, self).get_config()
        conf.update(self.pc.get_config())
        conf.update(self.ln.get_config())
        conf.update(self.mha.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf
class RBlock(tf.keras.layers.Layer):
    def __init__(self,input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 head_size=144,
                 num_heads=4,
                 kernel_size=32,
                 name="RBlock",
                 **kwargs):
        super(RBlock, self).__init__(name=name)
        self.ffm1 = FFModule(input_dim=input_dim,
                             dropout=dropout, fc_factor=fc_factor,
                             name="ff_module_1")
        self.mhsam = RMHSAModule(head_size=head_size, num_heads=num_heads,
                                dropout=dropout)
        self.convm = ConvModule(input_dim=input_dim, kernel_size=kernel_size,
                                dropout=dropout)
        self.ffm2 = FFModule(input_dim=input_dim,
                             dropout=dropout, fc_factor=fc_factor,
                             name="ff_module_2")
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, inputs, enc,training=False, **kwargs):
        outputs = self.ffm1(inputs, training=training)
        outputs = self.mhsam(outputs,enc, training=training)
        outputs = self.convm(outputs, training=training)
        outputs = self.ffm2(outputs, training=training)
        outputs = self.ln(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(RBlock, self).get_config()
        conf.update(self.ffm1.get_config())
        conf.update(self.mhsam.get_config())
        conf.update(self.convm.get_config())
        conf.update(self.ffm2.get_config())
        conf.update(self.ln.get_config())
        return conf
class Translator(tf.keras.Model):
    def __init__(self,inp_classes,
                 tar_classes,
                 dmodel=144,
                 num_blocks=16,
                 head_size=36,
                 num_heads=4,
                 fc_factor=0.5,
                 dropout=0.0,
                 kernel_size=32,
                 **kwargs):
        super(Translator, self).__init__()
        self.dmodel = dmodel
        self.decode_layers=[]
        for i in range(num_blocks):
            r_block = RBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                name=f"decoder_conformer_block_{i}"
            )
            self.decode_layers.append(r_block)
        self.inp_embedding=tf.keras.layers.Embedding(inp_classes,dmodel)
        self.fc = tf.keras.layers.Dense(units=tar_classes, activation="linear",
                                        use_bias=True, name="fully_connected")

    def _build(self):
        fake_a = tf.constant([[1, 2, 3, 4, 5, 6, 7]], tf.int32)
        fake_b = tf.random.uniform([1, 100, self.dmodel])
        self(fake_a,fake_b)


    def call(self, inputs,enc, training=None, mask=None):
        outputs=self.inp_embedding(inputs,training=training)
        for layer in self.decode_layers:
            outputs=layer(outputs,enc,training=training)
        outputs=self.fc(outputs,training=training)
        return outputs

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[
                     tf.TensorSpec([None, None], dtype=tf.int32),
                     tf.TensorSpec([None, None, 144], dtype=tf.float32),#TODO:根据自己的dmodel修改
                 ]
                 )
    def inference(self,inputs,enc):
        outputs = self.inp_embedding(inputs, training=False)
        for layer in self.decode_layers:
            outputs = layer(outputs, enc, training=False)
        outputs = self.fc(outputs, training=False)
        return outputs
class StreamingConformerEncoder(ConformerEncoder):
    def add_chunk_size(self,chunk_size,mel_size,hop_size):
        self.chunk_size=chunk_size
        self.mel_size=mel_size
        self.mel_length=self.chunk_size//hop_size if self.chunk_size%hop_size==0 else self.chunk_size//hop_size+1
        print(self.chunk_size,self.mel_size,self.mel_length)

    def call(self, inputs, training=False, **kwargs):

        if self.add_wav_info:

            B = tf.shape(inputs)[0]

            inputs = tf.reshape(inputs, [-1, self.chunk_size, 1])
            mel_inputs=self.mel_layer(inputs)
            mel_outputs = self.conv_subsampling(mel_inputs, training=training)
            wav_outputs = self.wav_layer(inputs, training=training)
            outputs = mel_outputs + wav_outputs
        else:
            B=tf.shape(inputs)[0]
            inputs = tf.reshape(inputs, [-1, self.chunk_size, 1])
            inputs = self.mel_layer(inputs)
            outputs = self.conv_subsampling(inputs, training=training)

        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, training=training)
        outputs = tf.reshape(outputs, [B, -1, self.dmodel])
        return outputs
    @tf.function(experimental_relax_shapes=True,
                 input_signature=[
                     tf.TensorSpec([None, None,1], dtype=tf.float32),
                 ]
                 )
    def inference(self, inputs, training=False, **kwargs):
        if self.add_wav_info:
            mel_inputs=self.mel_layer(inputs)
            mel_outputs = self.conv_subsampling(mel_inputs, training=training)
            wav_outputs = self.wav_layer(inputs, training=training)
            outputs = mel_outputs+wav_outputs
        else:
            inputs = self.mel_layer(inputs)
            outputs = self.conv_subsampling(inputs, training=training)

        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, training=training)
        return outputs