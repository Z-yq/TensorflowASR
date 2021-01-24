
import tensorflow as tf


from AMmodel.Multi_task_wrap import MultiTaskCTC
from AMmodel.wav_model import WavePickModel
from utils.tools import merge_two_last_dims
from AMmodel.layers.positional_encoding import PositionalEncoding
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
        self.bn = tf.keras.layers.BatchNormalization()
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
                 hop_size=80,
                 name="conformer_encoder",
                 **kwargs):
        super(ConformerEncoder, self).__init__(name=name, **kwargs)
        self.dmodel=dmodel
        self.reduction_factor=reduction_factor
        self.conv_subsampling = ConvSubsampling(
            odim=dmodel, reduction_factor=reduction_factor,
            dropout=dropout
        )
        self.conformer_blocks = []
        self.add_wav_info=add_wav_info
        if self.add_wav_info:
            self.wav_layer=WavePickModel(dmodel,hop_size)
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
        if self.add_wav_info:
            mel_inputs, wav_inputs = inputs
            mel_outputs = self.conv_subsampling(mel_inputs, training=training)
            wav_outputs = self.wav_layer(wav_inputs, training=training)
            outputs = mel_outputs+wav_outputs
        else:
            outputs = self.conv_subsampling(inputs, training=training)
        encoder_outputs=[]
        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, training=training)
            encoder_outputs.append(outputs)
        return encoder_outputs

    def get_config(self):
        conf = super(ConformerEncoder, self).get_config()
        conf.update(self.conv_subsampling.get_config())
        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf

class ConformerEncoder_(tf.keras.Model):
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
                 hop_size=80,
                 name="conformer_encoder",
                 **kwargs):
        super(ConformerEncoder_, self).__init__(name=name, **kwargs)
        self.dmodel=dmodel
        self.reduction_factor=reduction_factor
        self.conv_subsampling = ConvSubsampling(
            odim=dmodel, reduction_factor=reduction_factor,
            dropout=dropout
        )
        self.conformer_blocks = []
        self.add_wav_info=add_wav_info
        if self.add_wav_info:
            self.wav_layer=WavePickModel(dmodel,hop_size)
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
        if self.add_wav_info:
            mel_inputs, wav_inputs,enc_inputs = inputs
            mel_outputs = self.conv_subsampling(mel_inputs, training=training)
            wav_outputs = self.wav_layer(wav_inputs, training=training)
            outputs = mel_outputs+wav_outputs+enc_inputs
        else:
            mel_inputs,enc_inputs=inputs
            outputs = self.conv_subsampling(mel_inputs, training=training)
            outputs+=enc_inputs
        encoder_outputs=[]
        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, training=training)
            encoder_outputs.append(outputs)
        return encoder_outputs

    def get_config(self):
        conf = super(ConformerEncoder_, self).get_config()
        conf.update(self.conv_subsampling.get_config())
        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf

class ConformerMultiTaskCTC(MultiTaskCTC):
    def __init__(self,
                 config,

                 speech_config=dict,
                 **kwargs,
                 ):

        super(ConformerMultiTaskCTC,self).__init__(
            encoder1=ConformerEncoder(
                dmodel=config['dmodel'],
                reduction_factor=config['reduction_factor'],
                num_blocks=config['num_blocks'],
                head_size=config['head_size'],
                num_heads=config['num_heads'],
                kernel_size=config['kernel_size'],
                fc_factor=config['fc_factor'],
                dropout=config['dropout'],
                add_wav_info=speech_config['add_wav_info'],
                hop_size=int(speech_config['stride_ms'] * speech_config['sample_rate'] // 1000) * config['reduction_factor'],
                name=config['name']+'_1',
            ),
            encoder2=ConformerEncoder_(
                dmodel=config['dmodel'],
                reduction_factor=config['reduction_factor'],
                num_blocks=config['num_blocks'],
                head_size=config['head_size'],
                num_heads=config['num_heads'],
                kernel_size=config['kernel_size'],
                fc_factor=config['fc_factor'],
                dropout=config['dropout'],
                add_wav_info=speech_config['add_wav_info'],
                hop_size=int(speech_config['stride_ms'] * speech_config['sample_rate'] // 1000) * config[
                    'reduction_factor'],
                name=config['name']+'_2'
            ),
            encoder3=ConformerEncoder_(
                dmodel=config['dmodel'],
                reduction_factor=config['reduction_factor'],
                num_blocks=config['num_blocks'],
                head_size=config['head_size'],
                num_heads=config['num_heads'],
                kernel_size=config['kernel_size'],
                fc_factor=config['fc_factor'],
                dropout=config['dropout'],
                add_wav_info=speech_config['add_wav_info'],
                hop_size=int(speech_config['stride_ms'] * speech_config['sample_rate'] // 1000) * config[
                    'reduction_factor'],
                name=config['name']+'_3'
            ),


            classes1=config['classes1'],classes2=config['classes2'],classes3=config['classes3'],
            speech_config=speech_config,
            dmodel=config['dmodel']

        )
        self.time_reduction_factor = config['reduction_factor']
