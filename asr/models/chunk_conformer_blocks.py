import tensorflow as tf
from asr.models.wav_model import WavePickModel
from utils.tools import merge_two_last_dims
from asr.models.layers.time_frequency import Spectrogram, Melspectrogram
from leaf_audio import frontend


class GLU(tf.keras.layers.Layer):
    def __init__(self, axis=-1, name="glu_activation", **kwargs):
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
    def __init__(
            self,
            odim: int,
            chunk_num=16,
            reduction_factor: int = 4,
            dropout: float = 0.0,
            name="conv_subsampling",
            padding="valid",
            **kwargs,
    ):
        super(ConvSubsampling, self).__init__(name=name, **kwargs)
        assert reduction_factor % 2 == 0, "reduction_factor must be divisible by 2"
        self.T = chunk_num // reduction_factor
        self.conv1 = tf.keras.layers.Conv2D(
            filters=odim,
            kernel_size=(3, 3),
            strides=((reduction_factor // 2), 2),
            padding=padding,
            activation="relu",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=odim,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=padding,
            activation="relu",
        )
        self.linear = tf.keras.layers.Dense(odim)
        self.do = tf.keras.layers.Dropout(dropout)
        self.padding = padding

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        # B=tf.shape(inputs)[0]
        outputs = inputs
        # outputs=tf.concat([tf.zeros([B,4,80,1]),inputs],axis=1)
        if self.padding == 'valid':
            outputs = tf.pad(outputs, [[0, 0], [4, 0], [2, 2], [0, 0]])
        outputs = self.conv1(outputs, training=training)

        # if self.padding=='valid':
        # outputs=tf.pad(outputs,[[0,0],[2,0],[1,1],[0,0]])
        outputs = self.conv2(outputs, training=training)
        # print(outputs[:,-1])
        outputs = merge_two_last_dims(outputs)
        outputs = self.linear(outputs, training=training)
        return self.do(outputs, training=training)

    def stream_call(self, inputs, sub_cache, training=False):
        # print(inputs.shape,cache.shape)

        new_cache_1 = tf.concat([sub_cache, inputs], axis=1)
        outputs = new_cache_1
        if self.padding == 'valid':
            outputs = tf.pad(outputs, [[0, 0], [0, 0], [2, 2], [0, 0]])

        outputs = self.conv1(outputs, training=training)
        # new_cache_2=tf.concat([sub_cache_2,outputs],axis=1)
        # outputs=new_cache_2
        # if self.padding=='valid':
        # outputs=tf.pad(outputs,[[0,0],[0,0],[1,1],[0,0]])
        outputs = self.conv2(outputs, training=training)
        # print(outputs[:,-1])
        outputs = outputs[:, -self.T:]
        outputs = merge_two_last_dims(outputs)
        outputs = self.linear(outputs, training=training)
        # new_cache=tf.concat([new_cache_1[tf.newaxis],new_cache_2[tf.newaxis]],axis=0)
        return self.do(outputs, training=training), new_cache_1

    def get_config(self):
        conf = super(ConvSubsampling, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.conv2.get_config())
        conf.update(self.linear.get_config())
        conf.update(self.do.get_config())
        return conf


class FFModule(tf.keras.layers.Layer):
    def __init__(
            self, input_dim, dropout=0.0, fc_factor=0.5, name="ff_module", **kwargs
    ):
        super(FFModule, self).__init__(name=name, **kwargs)
        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization()
        self.ffn1 = tf.keras.layers.Dense(4 * input_dim)
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name="swish_activation"
        )
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


class ChunkMHSAModule(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, dropout=0.0, win_front=6, win_back=3, name="mhsa_module", **kwargs):
        super(ChunkMHSAModule, self).__init__(name=name, **kwargs)
        # self.pc = PositionalEncoding()
        self.ln = tf.keras.layers.LayerNormalization()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()
        self.win_front = win_front
        self.win_back = win_back
        self.d_model = num_heads * head_size

    def init_cache(self, B):
        return tf.zeros([B, 0, self.d_model])

    def _compute_chunk_mask(self, query, win_front, win_back):
        # win_front-=1
        q_seq_length = tf.shape(query)[1]
        B = tf.shape(query)[0]
        indexs = tf.expand_dims(tf.range(q_seq_length), -1)
        mask = tf.expand_dims(tf.range(q_seq_length), 0)
        mask = tf.repeat(mask, q_seq_length, axis=0)

        low = tf.nn.relu(indexs - win_front)
        high = tf.clip_by_value(indexs + win_back, 0, q_seq_length)

        low = low - tf.nn.relu(low - q_seq_length + win_back)
        high = high + tf.nn.relu(win_back - high)
        mask = tf.cast(mask < low, tf.float32) + tf.cast(mask > high, tf.float32)
        mask = tf.expand_dims(mask, 0)
        mask = tf.repeat(mask, B, 0)
        mask = tf.where(mask == 0., 1., 0.)
        mask = tf.cast(mask, tf.bool)
        return mask

    def _compute_causal_mask(self, query, value=None):
        q_seq_length = tf.shape(query)[1]

        v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
        mask = tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0
        )
        mask = tf.where(tf.cast(mask, tf.float32) == 1.0, 0.0, 1.0)
        return mask

    def _compute_mask(self, query, win_front=26, win_back=3):
        mask = self._compute_causal_mask(query) + self._compute_chunk_mask(
            query, win_front, win_back,
        )
        mask = tf.clip_by_value(mask, 0.0, 1.0)
        return mask

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, mask=None, training=False, **kwargs):
        # outputs = self.pc(inputs)
        mask = self._compute_chunk_mask(inputs, self.win_front, self.win_back)
        outputs = self.ln(inputs, training=training)
        outputs = self.mha(outputs, outputs, attention_mask=mask, training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def stream_call(self, inputs, cache, mask=None, training=False):
        T = tf.shape(inputs)[1]
        # print(T)
        new_cache = tf.concat([cache, inputs], axis=1)
        outputs = self.ln(new_cache, training=training)
        mask = self._compute_chunk_mask(outputs, self.win_front, self.win_back)
        query = outputs[:, -T:]
        mask = mask[:, -T:]
        # tf.print('mha T:',T)
        outputs = self.mha(query, outputs, training=training, attention_mask=mask, use_causal_mask=False)

        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs, new_cache

    def get_config(self):
        conf = super(ChunkMHSAModule, self).get_config()
        # conf.update(self.pc.get_config())
        conf.update(self.ln.get_config())
        conf.update(self.mha.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf


class ChunkConvModule(tf.keras.layers.Layer):
    def __init__(
            self,
            input_dim,
            kernel_size=32,
            dropout=0.0,
            name="conv_module",
            T=4,
            padding="same",
            **kwargs,
    ):
        super(ChunkConvModule, self).__init__(name=name, **kwargs)
        self.T = T
        self.ln = tf.keras.layers.LayerNormalization()
        self.pw_conv_1 = tf.keras.layers.Conv1D(
            filters=2 * input_dim,
            kernel_size=1,
            strides=1,
            padding=padding,
            name="pw_conv_1",
        )
        self.glu = GLU()
        self.dw_conv = tf.keras.layers.SeparableConv1D(
            filters=2 * input_dim,
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            depth_multiplier=1,
            name="dw_conv",
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name="swish_activation"
        )
        self.pw_conv_2 = tf.keras.layers.Conv1D(
            filters=input_dim,
            kernel_size=1,
            strides=1,
            padding=padding,
            name="pw_conv_2",
        )
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()
        self.kernel_size = kernel_size
        self.d_model = input_dim

    def init_cache(self, B):
        return tf.zeros([B, 0, self.d_model])

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

    def stream_call(self, inputs, cache, training=False, **kwargs):
        # print(T)
        T = tf.shape(inputs)[1]
        new_cache = tf.concat([cache, inputs], axis=1)
        outputs = self.ln(new_cache, training=training)
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = self.do(outputs, training=training)
        # print(outputs.shape)
        outputs = outputs[:, -T:]
        # tf.print('cnn T:',T,self.T)
        outputs = self.res_add([inputs, outputs])

        return outputs, new_cache

    def get_config(self):
        conf = super(ChunkConvModule, self).get_config()
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


class ChunkConformerBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            input_dim,
            dropout=0.0,
            fc_factor=0.5,
            head_size=144,
            num_heads=4,
            kernel_size=32,
            name="conformer_block",
            padding="causal",
            win_front=6,
            win_back=2,
            **kwargs,
    ):
        super(ChunkConformerBlock, self).__init__(name=name, **kwargs)
        self.ffm1 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            name="ff_module_1",
        )
        self.mhsam = ChunkMHSAModule(
            head_size=head_size, num_heads=num_heads, dropout=dropout, win_front=win_front, win_back=win_back,
        )
        self.convm = ChunkConvModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            padding=padding,
            T=4 + win_back,
        )
        self.ffm2 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            name="ff_module_2",
        )
        self.ln = tf.keras.layers.LayerNormalization()

    def init_cache(self, B):
        mha_cache = self.mhsam.init_cache(B)
        cnn_cache = self.convm.init_cache(B)
        return mha_cache, cnn_cache

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, mask=None, training=False, **kwargs):
        outputs = self.ffm1(inputs, training=training)
        outputs = self.mhsam(outputs, mask=mask, training=training)
        outputs = self.convm(outputs, training=training)
        outputs = self.ffm2(outputs, training=training)
        outputs = self.ln(outputs, training=training)
        return outputs

    def stream_call(self, inputs, mha_cache, cnn_cache, mask=None, training=False, **kwargs):
        outputs = self.ffm1(inputs, training=training)
        outputs, new_mha_cache = self.mhsam.stream_call(outputs, mha_cache, mask=mask, training=training)
        outputs, new_cnn_cache = self.convm.stream_call(outputs, cnn_cache, training=training)
        outputs = self.ffm2(outputs, training=training)
        outputs = self.ln(outputs, training=training)
        # new_cache=tf.concat([mhsa_cache[:,tf.newaxis],cnn_cache[:,tf.newaxis]],axis=1)
        return outputs, new_mha_cache, new_cnn_cache

    def get_config(self):
        conf = super(ChunkConformerBlock, self).get_config()
        conf.update(self.ffm1.get_config())
        conf.update(self.mhsam.get_config())
        conf.update(self.convm.get_config())
        conf.update(self.ffm2.get_config())
        conf.update(self.ln.get_config())
        return conf


class ChunkConformerFront(tf.keras.Model):
    def __init__(self, dmodel=144,
                 reduction_factor=4,
                 dropout=0.0,
                 sample_rate=16000,
                 n_mels=80,
                 mel_layer_trainable=False,
                 stride_ms=10,
                 chunk_num=16,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hop_size = int(stride_ms * sample_rate // 1000) * reduction_factor

        self.reduction_factor = reduction_factor
        self.conv_subsampling = ConvSubsampling(
            odim=dmodel, reduction_factor=reduction_factor, dropout=dropout, padding='valid', chunk_num=chunk_num
        )
        self.n_mels = n_mels
        self.chunk_num = chunk_num
        self.wav_buf_length = int(chunk_num * stride_ms * sample_rate // 1000)
        self.dmodel = dmodel
        self.mel_layer = Melspectrogram(
            sr=sample_rate,
            n_mels=n_mels,
            n_hop=int(stride_ms * sample_rate // 1000),
            n_dft=1024,
            trainable_fb=mel_layer_trainable,
            padding='valid',
        )
        self.sub_length=int(self.chunk_num // self.reduction_factor)
        self.mel_layer.trainable = mel_layer_trainable

    def _build(self, ):
        self(tf.zeros([1, 16000, 1]))

    def init_caches(self, B):
        return tf.zeros([B, 0, 1]), tf.zeros([B, self.sub_length, self.n_mels,
                                              1])  # ,tf.zeros([B,self.chunk_num//self.reduction_factor//2,self.n_mels//2,self.dmodel])

    # @tf.function
    def call(self, inputs, training=False):
        outputs = self.mel_layer(inputs, training=training)

        outputs = self.conv_subsampling(outputs, training=training)

        return outputs

    def stream_call(self, inputs, wav_cache, sub_cache):
        training = False
        new_wav_cache = tf.concat([wav_cache, inputs], axis=1)

        outputs = self.mel_layer(new_wav_cache, training=training)

        outputs = outputs[:, -self.chunk_num:]
        outputs, new_sub_cache = self.conv_subsampling.stream_call(outputs, sub_cache, training=training)

        new_wav_cache = new_wav_cache[:, -self.wav_buf_length:]
        new_sub_cache = new_sub_cache[:, -self.sub_length:]
        return outputs, new_wav_cache, new_sub_cache



class ChunkConformerEncoder(tf.keras.Model):
    def __init__(
            self,
            dmodel=144,
            num_blocks=16,
            head_size=36,
            num_heads=4,
            kernel_size=32,
            fc_factor=0.5,
            dropout=0.0,
            win_front=21,
            win_back=4,
            name="conformer_encoder",
            padding="causal",
            **kwargs,
    ):
        super(ChunkConformerEncoder, self).__init__(name=name, **kwargs)
        self.dmodel = dmodel
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.fc_factor = fc_factor
        self.dropout = dropout
        self.head_size = head_size
        self.win_back = win_back
        self.win_front = win_front
        self.kernel_size = kernel_size
        self.conformer_blocks = []
        for i in range(num_blocks):
            conformer_block = ChunkConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                win_back=win_back,
                win_front=win_front,
                name=f"chunk_conformer_block_{i}",
                padding=padding,
            )
            self.conformer_blocks.append(conformer_block)

    def _build(self):
        fake = tf.random.uniform([1, 10, self.dmodel])
        self(fake)

    # @tf.function
    def init_caches(self, B):
        new_mha_caches = []
        new_cnn_caches = []
        for layer in self.conformer_blocks:
            mha_cache, cnn_cache = layer.init_cache(B)
            new_mha_caches.append(mha_cache[tf.newaxis])
            new_cnn_caches.append(cnn_cache[tf.newaxis])
        new_mha_caches = tf.concat(new_mha_caches, axis=0)
        new_cnn_caches = tf.concat(new_cnn_caches, axis=0)

        return new_mha_caches, new_cnn_caches

    # @tf.function
    def call(self, inputs, mask=None, training=False, **kwargs):
        outputs = inputs

        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, mask=mask, training=training)

        return outputs

    def stream_call(self, inputs, mha_caches, cnn_caches, **kwargs):
        training = False
        new_mha_caches = []
        new_cnn_caches = []
        outputs = inputs

        for idx, cblock in enumerate(self.conformer_blocks):
            mha_cache = mha_caches[idx]
            cnn_cache = cnn_caches[idx]
            outputs, new_mha_cache, new_cnn_cache = cblock.stream_call(outputs, mha_cache, cnn_cache, training=training)
            new_mha_caches.append(new_mha_cache[tf.newaxis])
            new_cnn_caches.append(new_cnn_cache[tf.newaxis])
        new_mha_caches = tf.concat(new_mha_caches, axis=0)
        new_cnn_caches = tf.concat(new_cnn_caches, axis=0)
        # return outputs,new_mha_caches,new_cnn_caches
        if self.win_back != 0:
            valid_outputs = outputs[:, :-self.win_back]
            valid_new_mha_caches = new_mha_caches[:, :, :-self.win_back]
            valid_new_cnn_caches = new_cnn_caches[:, :, :-self.win_back]
            valid_new_mha_caches = valid_new_mha_caches[:, :, -self.win_front:]
            valid_new_cnn_caches = valid_new_cnn_caches[:, :, -self.kernel_size:]
            unvalid_outputs = outputs[:, -self.win_back:]
        else:
            valid_outputs = outputs
            valid_new_mha_caches = new_mha_caches
            valid_new_cnn_caches = new_cnn_caches
            valid_new_mha_caches = valid_new_mha_caches[:, :, -self.win_front:]
            valid_new_cnn_caches = valid_new_cnn_caches[:, :, -self.kernel_size:]
            unvalid_outputs = tf.zeros_like([1])

        return valid_outputs, valid_new_mha_caches, valid_new_cnn_caches, unvalid_outputs

    @tf.function(input_signature=[
        tf.TensorSpec([1, None, 144], tf.float32),
        tf.TensorSpec([15, 1, None, 144], tf.float32),
        tf.TensorSpec([15, 1, None, 144], tf.float32),

    ], experimental_relax_shapes=True)
    def onnx_convert(self, inputs, mha_caches, cnn_caches, **kwargs):
        return self.stream_call(inputs, mha_caches, cnn_caches)

    def get_config(self):
        conf = super(ChunkConformerEncoder, self).get_config()

        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf


class ChunkCTCDecoder(tf.keras.Model):
    def __init__(
            self,
            num_classes,
            dmodel=144,
            num_blocks=16,
            head_size=36,
            num_heads=4,
            fc_factor=0.5,
            dropout=0.0,
            kernel_size=32,
            win_front=6,
            win_back=2,
            name="ctc_decoder",
            **kwargs,
    ):
        super(ChunkCTCDecoder, self).__init__()
        self.decode_layers = []
        self.num_blocks = num_blocks
        self.dmodel = dmodel
        self.project = tf.keras.layers.Dense(dmodel)
        self.win_back = win_back
        self.win_front = win_front
        self.kernel_size = kernel_size
        for i in range(num_blocks):
            conformer_block = ChunkConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                win_back=win_back,
                win_front=win_front,
                name=name + f"ChunkCTCdecoder_conformer_block_{i}",
            )
            self.decode_layers.append(conformer_block)

        self.fc = tf.keras.layers.Dense(
            units=num_classes,
            activation="linear",
            use_bias=True,
            name=name + "fully_connected",
        )

    def _build(self):
        fake = tf.random.uniform([1, 10, self.dmodel])
        self(fake)

    def init_caches(self, B):
        new_mha_caches = []
        new_cnn_caches = []
        for layer in self.decode_layers:
            mha_cache, cnn_cache = layer.init_cache(B)
            new_mha_caches.append(mha_cache[tf.newaxis])
            new_cnn_caches.append(cnn_cache[tf.newaxis])
        new_mha_caches = tf.concat(new_mha_caches, axis=0)
        new_cnn_caches = tf.concat(new_cnn_caches, axis=0)
        return new_mha_caches, new_cnn_caches

    # @tf.function
    def call(self, inputs, training=None, mask=None):
        outputs = self.project(inputs, training=training)
        for layer in self.decode_layers:
            outputs = layer(outputs, mask=mask, training=training)
        ctc_outs = self.fc(outputs, training=training)
        return ctc_outs, outputs

    # def stream_call(self,inputs,mha_caches,cnn_caches):

    def stream_call(self, inputs, mha_caches, cnn_caches, **kwargs):
        training = False
        new_mha_caches = []
        new_cnn_caches = []
        outputs = self.project(inputs, training=training)

        for idx, cblock in enumerate(self.decode_layers):
            mha_cache = mha_caches[idx]
            cnn_cache = cnn_caches[idx]
            outputs, new_mha_cache, new_cnn_cache = cblock.stream_call(outputs, mha_cache, cnn_cache, training=training)
            new_mha_caches.append(new_mha_cache[tf.newaxis])
            new_cnn_caches.append(new_cnn_cache[tf.newaxis])
        new_mha_caches = tf.concat(new_mha_caches, axis=0)
        new_cnn_caches = tf.concat(new_cnn_caches, axis=0)
        ctc_outs = self.fc(outputs, training=training)

        if self.win_back != 0:
            valid_outputs = ctc_outs[:, :-self.win_back]
            valid_feature = outputs[:, :-self.win_back]
            valid_new_mha_caches = new_mha_caches[:, :, :-self.win_back]
            valid_new_cnn_caches = new_cnn_caches[:, :, :-self.win_back]
            valid_new_mha_caches = valid_new_mha_caches[:, :, -self.win_front:]
            valid_new_cnn_caches = valid_new_cnn_caches[:, :, -self.kernel_size:]
            unvalid_outputs = ctc_outs[:, -self.win_back:]
        else:
            valid_outputs = ctc_outs
            valid_feature = outputs
            valid_new_mha_caches = new_mha_caches
            valid_new_cnn_caches = new_cnn_caches
            valid_new_mha_caches = valid_new_mha_caches[:, :, -self.win_front:]
            valid_new_cnn_caches = valid_new_cnn_caches[:, :, -self.kernel_size:]
            unvalid_outputs = tf.zeros_like(valid_outputs)
        return valid_outputs, valid_feature, valid_new_mha_caches, valid_new_cnn_caches, unvalid_outputs

    @tf.function(input_signature=[
        tf.TensorSpec([1, None, 144], tf.float32),
        tf.TensorSpec([1, 1, None, 144], tf.float32),
        tf.TensorSpec([1, 1, None, 144], tf.float32),
    ], experimental_relax_shapes=True)
    def onnx_convert(self, inputs, mha_caches, cnn_caches):
        return self.stream_call(self, inputs, mha_caches, cnn_caches)
        # return outputs,new_mha_caches,new_cnn_caches




class ContextHelper(tf.keras.Model):
    def __init__(
            self,
            num_classes,
            dmodel=144,
            num_blocks=16,
            head_size=36,
            num_heads=4,
            fc_factor=0.5,
            dropout=0.0,
            kernel_size=32,
            win_front=36,
            win_back=0,
            name="ContextHelper",
            **kwargs,
    ):
        super(ContextHelper, self).__init__()
        self.decode_layers = []

        self.cross_layers = []
        self.dmodel = dmodel
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.win_front = win_front
        self.win_back = win_back
        self.num_blocks = num_blocks
        # self.project = tf.keras.layers.Dense(dmodel)
        self.sample_helper = tf.keras.layers.Embedding(num_classes, dmodel)
        for i in range(num_blocks):
            conformer_block = ChunkConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                win_front=win_front,
                win_back=win_back,
                name=name + f"_decoder_block_{i}",
            )
            self.decode_layers.append(conformer_block)

        self.num_blocks = num_blocks

    def _build(self):
        fake_ctc = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]])
        self.phone_call(fake_ctc)

    def init_caches(self, B):
        new_mha_caches = []
        new_cnn_caches = []
        for layer in self.decode_layers:
            mha_cache, cnn_cache = layer.init_cache(B)
            new_mha_caches.append(mha_cache[tf.newaxis])
            new_cnn_caches.append(cnn_cache[tf.newaxis])
        new_mha_caches = tf.concat(new_mha_caches, axis=0)
        new_cnn_caches = tf.concat(new_cnn_caches, axis=0)
        return new_mha_caches, new_cnn_caches

    def call(self, outputs, training=False, mask=None):

        for block in self.decode_layers:
            outputs = block(outputs, mask=mask, training=training)

        return outputs

    def phone_call(self, label_gts, training=False):
        outputs = self.sample_helper(label_gts, training=training)
        logits = self(outputs, training=training)
        return outputs, logits

    def stream_call(self, outputs, mha_caches, cnn_caches):

        training = False
        new_mha_caches = []
        new_cnn_caches = []

        for idx, cblock in enumerate(self.decode_layers):
            mha_cache = mha_caches[idx]
            cnn_cache = cnn_caches[idx]
            outputs, new_mha_cache, new_cnn_cache = cblock.stream_call(outputs, mha_cache, cnn_cache, training=training)
            new_mha_caches.append(new_mha_cache[tf.newaxis])
            new_cnn_caches.append(new_cnn_cache[tf.newaxis])
        new_mha_caches = tf.concat(new_mha_caches, axis=0)
        new_cnn_caches = tf.concat(new_cnn_caches, axis=0)
        # return outputs,new_mha_caches,new_cnn_caches

        valid_outputs = outputs
        valid_new_mha_caches = new_mha_caches
        valid_new_cnn_caches = new_cnn_caches
        valid_new_mha_caches = valid_new_mha_caches[:, :, -self.win_front:]
        valid_new_cnn_caches = valid_new_cnn_caches[:, :, -self.kernel_size:]

        return valid_outputs, valid_new_mha_caches, valid_new_cnn_caches


class ChunkConformer(tf.keras.Model):
    def __init__(self, config, phone, txt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config = config['model_config']
        self.model_config['ChunkCTCPicker']['num_classes'] = phone
        self.model_config['ContextHelper']['num_classes'] = phone
        self.model_config['ChunkCTCDecoder']['num_classes'] = txt
        self.front = ChunkConformerFront(**self.model_config['ChunkConformerFront'])
        self.encoder = ChunkConformerEncoder(**self.model_config['ChunkConformerEncoder'])
        self.phone_picker = ChunkCTCDecoder(**self.model_config['ChunkCTCPicker'])
        self.decoder = ChunkCTCDecoder(**self.model_config['ChunkCTCDecoder'])
        self.helper = ContextHelper(**self.model_config['ContextHelper'])
        self.phone_num_classes = phone
        self.txt_num_classes = txt
        self.dmodel = self.decoder.dmodel
        self._build()

    def _build(self):
        fake = tf.random.uniform([1, 16000, 1])

        self(fake)
        self.helper._build()

    def init_picker_caches(self, B):
        front_wav_cache, front_sub_cache = self.front.init_caches(B)
        encoder_mha_cache, encoder_cnn_cache = self.encoder.init_caches(B)
        decoder_mha_cache, decoder_cnn_cache = self.phone_picker.init_caches(B)
        dec_inp = tf.zeros([1, 0, self.encoder.dmodel])
        # cross_mha_caches,cross_cnn_caches,cross_query_key_caches=self.corrector.init_caches(B)
        return (
            front_wav_cache, front_sub_cache, encoder_mha_cache, encoder_cnn_cache, decoder_mha_cache,
            decoder_cnn_cache,
            dec_inp)

    def init_decoder_caches(self, B):
        helper_mha_cache, helper_cnn_cache = self.helper.init_caches(B)
        decoder_mha_cache, decoder_cnn_cache = self.decoder.init_caches(B)
        dec_inp = tf.zeros([1, 0, self.phone_picker.dmodel])
        return (helper_mha_cache, helper_cnn_cache, decoder_mha_cache, decoder_cnn_cache, dec_inp)

    def predict(self, x):
        front_out = self.front(x, training=False)
        enc_output = self.encoder(front_out, training=False)
        phone_outs, hidden_output = self.phone_picker(enc_output, training=False)
        picked_f, picked_c = self.feature_pick(hidden_output, phone_outs)
        help_out = self.helper(picked_f, training=False)
        decoder_outputs, _ = self.decoder(help_out, training=False)
        return decoder_outputs

    def stream_predict(self, input_wav, caches):
        front_wav_cache, front_sub_cache, encoder_mha_cache, encoder_cnn_cache, decoder_mha_cache, decoder_cnn_cache, dec_inp = caches
        front_out, front_wav_cache, front_sub_cache = self.front.stream_call(input_wav, front_wav_cache,
                                                                             front_sub_cache)
        valid_enc_out, encoder_mha_cache, encoder_cnn_cache, _ = self.encoder.stream_call(front_out, encoder_mha_cache,
                                                                                          encoder_cnn_cache)
        dec_inp = tf.concat([dec_inp, valid_enc_out], axis=1)
        valid_ctc_out, valid_hidden_out, decoder_mha_cache, decoder_cnn_cache, unvalid_ctc_out = self.phone_picker.stream_call(
            dec_inp, decoder_mha_cache, decoder_cnn_cache)
        T = tf.shape(valid_ctc_out)[1]
        dec_inp = dec_inp[:, T:]

        return valid_ctc_out, unvalid_ctc_out, valid_hidden_out, (
            front_wav_cache, front_sub_cache, encoder_mha_cache, encoder_cnn_cache, decoder_mha_cache,
            decoder_cnn_cache,
            dec_inp)

    def decoder_stream_predict(self, valid_enc_out, caches):
        helper_mha_cache, helper_cnn_cache, decoder_mha_cache, decoder_cnn_cache, dec_inp = caches
        valid_enc_out, helper_mha_cache, helper_cnn_cache = self.helper.stream_call(valid_enc_out, helper_mha_cache,
                                                                                    helper_cnn_cache)
        dec_inp = tf.concat([dec_inp, valid_enc_out], axis=1)
        valid_ctc_out, valid_hidden_out, decoder_mha_cache, decoder_cnn_cache, unvalid_ctc_out = self.decoder.stream_call(
            dec_inp, decoder_mha_cache, decoder_cnn_cache)
        T = tf.shape(valid_ctc_out)[1]
        dec_inp = dec_inp[:, T:]

        return valid_ctc_out, unvalid_ctc_out, (
            helper_mha_cache, helper_cnn_cache, decoder_mha_cache, decoder_cnn_cache, dec_inp)

    def picker_onnx_infer(self, input_wav, front_wav_cache, front_sub_cache, encoder_mha_cache, encoder_cnn_cache,
                          picker_mha_cache, picker_cnn_cache, dec_inp):
        front_out, front_wav_cache, front_sub_cache = self.front.stream_call(input_wav, front_wav_cache,
                                                                             front_sub_cache)
        valid_enc_out, encoder_mha_cache, encoder_cnn_cache, _ = self.encoder.stream_call(front_out, encoder_mha_cache,
                                                                                          encoder_cnn_cache)
        dec_inp = tf.concat([dec_inp, valid_enc_out], axis=1)
        valid_ctc_out, valid_hidden_out, picker_mha_cache, picker_cnn_cache, unvalid_ctc_out = self.phone_picker.stream_call(
            dec_inp, picker_mha_cache, picker_cnn_cache)
        T = tf.shape(valid_ctc_out)[1]
        dec_inp = dec_inp[:, T:]

        return valid_ctc_out, unvalid_ctc_out, valid_hidden_out, front_wav_cache, front_sub_cache, encoder_mha_cache, encoder_cnn_cache, picker_mha_cache, picker_cnn_cache, dec_inp

    def decoder_onnx_infer(self, valid_enc_out, helper_mha_cache, helper_cnn_cache, decoder_mha_cache,
                           decoder_cnn_cache, dec_inp):
        valid_enc_out, helper_mha_cache, helper_cnn_cache = self.helper.stream_call(valid_enc_out, helper_mha_cache,
                                                                                    helper_cnn_cache)
        dec_inp = tf.concat([dec_inp, valid_enc_out], axis=1)
        valid_ctc_out, _, decoder_mha_cache, decoder_cnn_cache, unvalid_ctc_out = self.decoder.stream_call(
            dec_inp, decoder_mha_cache, decoder_cnn_cache)
        T = tf.shape(valid_ctc_out)[1]
        dec_inp = dec_inp[:, T:]

        return valid_ctc_out, unvalid_ctc_out, helper_mha_cache, helper_cnn_cache, decoder_mha_cache, decoder_cnn_cache, dec_inp

    def picker_onnx_convert(self):
        self.picker_inp_sig = [
            tf.TensorSpec([1, None, 1], tf.float32),
            tf.TensorSpec([1, None, 1], tf.float32),
            tf.TensorSpec([1, None, self.front.n_mels, 1], tf.float32),
            tf.TensorSpec([self.encoder.num_blocks, 1, None, self.encoder.dmodel], tf.float32),
            tf.TensorSpec([self.encoder.num_blocks, 1, None, self.encoder.dmodel], tf.float32),
            tf.TensorSpec([self.phone_picker.num_blocks, 1, None, self.phone_picker.dmodel], tf.float32),
            tf.TensorSpec([self.phone_picker.num_blocks, 1, None, self.phone_picker.dmodel], tf.float32),
            tf.TensorSpec([1, None, self.phone_picker.dmodel], tf.float32),
        ]
        self.picker_inps = ['input_wav', 'front_wav_cache', 'front_sub_cache', 'encoder_mha_cache', 'encoder_cnn_cache',
                            'picker_mha_cache', 'picker_cnn_cache', 'dec_inp']
        self.picker_states = self.init_picker_caches(1)
        self.picker_states = [i.numpy() for i in self.picker_states]
        return tf.function(self.picker_onnx_infer, input_signature=self.picker_inp_sig)

    def decoder_onnx_convert(self):
        self.decoder_inp_sig = [
            tf.TensorSpec([1, None, self.phone_picker.dmodel], tf.float32),
            tf.TensorSpec([self.helper.num_blocks, 1, None, self.helper.dmodel], tf.float32),
            tf.TensorSpec([self.helper.num_blocks, 1, None, self.helper.dmodel], tf.float32),
            tf.TensorSpec([self.decoder.num_blocks, 1, None, self.decoder.dmodel], tf.float32),
            tf.TensorSpec([self.decoder.num_blocks, 1, None, self.decoder.dmodel], tf.float32),
            tf.TensorSpec([1, None, self.helper.dmodel], tf.float32),
        ]
        self.decoder_inps = [
            'valid_enc_out', 'helper_mha_cache', 'helper_cnn_cache', 'decoder_mha_cache', 'decoder_cnn_cache', 'dec_inp'
        ]
        self.decoder_states = self.init_decoder_caches(1)
        self.decoder_states = [i.numpy() for i in self.decoder_states]
        return tf.function(self.decoder_onnx_infer, input_signature=self.decoder_inp_sig)

    def feature_pick(self, encoder_hidden_states, ctc_outs, max_T=None):
        """Length regulator logic."""

        ctc_arg_out = tf.argmax(ctc_outs, -1)
        durations_gt = tf.where(ctc_arg_out != self.phone_num_classes - 1, 1, 0)
        durations_gt = tf.cast(durations_gt, tf.int32)
        sum_durations = tf.reduce_sum(durations_gt, axis=-1)  # [batch_size]
        max_durations = tf.reduce_max(sum_durations)
        if max_T is not None:
            # label_max_durations = tf.reduce_max(label_lengths)
            max_durations = tf.reduce_max([max_durations, max_T])
        # input_shape = tf.shape(encoder_hidden_states)
        batch_size = tf.shape(encoder_hidden_states)[0]
        hidden_size = tf.shape(encoder_hidden_states)[-1]
        ctc_size = tf.shape(ctc_outs)[-1]
        # initialize output hidden states and encoder masking.
        feature_outputs = tf.zeros(shape=[0, max_durations, hidden_size], dtype=tf.float32)
        ctc_outputs = tf.zeros(shape=[0, max_durations, ctc_size], dtype=tf.float32)

        def condition(i,
                      batch_size,
                      feature_outputs,
                      ctc_outputs,
                      # encoder_masks,
                      encoder_hidden_states,
                      ctc_outs,
                      durations_gt,
                      max_durations):
            return tf.less(i, batch_size)

        def body(i,
                 batch_size,
                 feature_outputs,
                 ctc_outputs,
                 # encoder_masks,
                 encoder_hidden_states,
                 ctc_outs,
                 durations_gt,
                 max_durations):
            repeats = durations_gt[i]
            real_length = tf.reduce_sum(repeats)
            pad_size = max_durations - real_length
            # masks = tf.sequence_mask([real_length], max_durations, dtype=tf.int32)
            repeat_encoder_hidden_states = tf.repeat(
                encoder_hidden_states[i],
                repeats=repeats,
                axis=0
            )
            repeat_ctc_outs = tf.repeat(
                ctc_outs[i],
                repeats=repeats,
                axis=0
            )
            repeat_encoder_hidden_states = tf.expand_dims(
                tf.pad(
                    repeat_encoder_hidden_states, [[0, pad_size], [0, 0]]
                ),
                0)  # [1, max_durations, hidden_size]
            repeat_ctc_outs = tf.expand_dims(
                tf.pad(
                    repeat_ctc_outs, [[0, pad_size], [0, 0]]
                ),
                0)  # [1, max_durations, hidden_size]
            feature_outputs = tf.concat([feature_outputs, repeat_encoder_hidden_states], axis=0)
            ctc_outputs = tf.concat([ctc_outputs, repeat_ctc_outs], axis=0)
            # encoder_masks = tf.concat([encoder_masks, masks], axis=0)
            return [i + 1, batch_size, feature_outputs, ctc_outputs,
                    encoder_hidden_states, ctc_outs, durations_gt, max_durations]

        # initialize iteration i.
        i = tf.constant(0, dtype=tf.int32)
        _, _, feature_outputs, ctc_outputs, _, _, _, _ = tf.while_loop(
            condition,
            body,
            [i, batch_size, feature_outputs, ctc_outputs, encoder_hidden_states, ctc_outs, durations_gt, max_durations],
            shape_invariants=[i.get_shape(),
                              batch_size.get_shape(),
                              tf.TensorShape([None, None, self.dmodel]),
                              tf.TensorShape([None, None, self.phone_num_classes]),
                              # tf.TensorShape([None, None]),
                              encoder_hidden_states.get_shape(),
                              ctc_outs.get_shape(),
                              durations_gt.get_shape(),
                              max_durations.get_shape()],

        )
        return feature_outputs, ctc_outputs

    def call(self, inputs, training=False):
        front_out = self.front(inputs, training=training)
        enc_output = self.encoder(front_out, training=training)
        _, hidden_output = self.phone_picker(enc_output, training=training)
        # picked_f,picked_c=self.feature_pick(hidden_output,decoder_output)
        final_outputs = self.decoder(hidden_output, training=training)
        return final_outputs

    def ctc_acc(self, labels, y_pred):
        T1 = tf.shape(y_pred)[1]
        T2 = tf.shape(labels)[1]
        T = tf.reduce_min([T1, T2])
        y_pred = y_pred[:, :T]
        labels = labels[:, :T]

        mask = tf.cast(tf.not_equal(labels, 0), 1.0)
        y_pred = tf.cast(y_pred, tf.float32)
        labels = tf.cast(labels, tf.float32)

        value = tf.cast(labels == y_pred, tf.float32)

        accs = tf.reduce_sum(value * mask, -1) / (tf.reduce_sum(mask, -1) + 1e-6)
        return accs

    def masked_mse(self, y_pred, y_true, label_gts):
        need = tf.cast(tf.where(label_gts != 0, 1, 0), tf.float32)

        loss = tf.losses.mse(y_pred, y_true)
        need_loss = tf.reduce_sum(loss * need, -1, keepdims=True) / (
                tf.reduce_sum(need, -1, keepdims=True) + 1e-6
        )
        return need_loss

    def train_step(self, batch):
        features, input_length, phone_labels, phone_label_length, txt_label, txt_label_lengths, extra_phones, extra_phone_length, extra_txts, extra_txt_length = \
            batch[0]
        # tf.print(tf.shape(features),tf.shape(phone_labels))
        # sampler_labels=tf.concat([tf.ones_like(phone_labels,tf.int32)[:,:1],phone_labels])

        max_T = tf.shape(phone_labels)[1]
        with tf.GradientTape() as tape:
            front_out = self.front(features, training=True)
            enc_output = self.encoder(front_out, training=True)
            phone_output, hidden_outputs = self.phone_picker(enc_output, training=True)
            picked_f, picked_c = self.feature_pick(hidden_outputs, phone_output, max_T)
            _, helper_out = self.helper.phone_call(extra_phones, training=True)
            picke_help = self.helper(picked_f, training=True)

            txt_output, _ = self.decoder(picke_help, training=True)
            help_output, _ = self.decoder(helper_out, training=True)

            T = tf.shape(picked_f)[1]
            new_input_length = tf.ones_like(input_length, tf.int32) * T

            phone_output = tf.nn.softmax(phone_output, -1)
            txt_output = tf.nn.softmax(txt_output, -1)
            help_output = tf.nn.softmax(help_output, -1)
            phone_ctc_loss = tf.keras.backend.ctc_batch_cost(
                tf.cast(phone_labels, tf.int32),
                tf.cast(phone_output, tf.float32),
                tf.cast(input_length[:, tf.newaxis], tf.int32),
                tf.cast(phone_label_length[:, tf.newaxis], tf.int32),
            )
            txt_ctc_loss = tf.keras.backend.ctc_batch_cost(
                tf.cast(txt_label, tf.int32),
                tf.cast(txt_output, tf.float32),
                tf.cast(new_input_length[:, tf.newaxis], tf.int32),
                tf.cast(txt_label_lengths[:, tf.newaxis], tf.int32),
            )

            help_ctc_loss = tf.keras.backend.ctc_batch_cost(
                tf.cast(extra_txts, tf.int32),
                tf.cast(help_output, tf.float32),
                tf.cast(extra_phone_length[:, tf.newaxis], tf.int32),
                tf.cast(extra_txt_length[:, tf.newaxis], tf.int32),
            )
            # mask_mse=self.maske_mse(picked_f,picke_help,phone_labels)
            # loss_mask=tf.math.is_inf(ctc_loss_final)
            # loss_mask=tf.where(loss_mask==True,0.,1.0)
            # masked_ctc_final_loss=tf.clip_by_value(ctc_loss_final,0.,1000.)
            train_loss = phone_ctc_loss + txt_ctc_loss + help_ctc_loss

        self.optimizer.minimize(train_loss, self.trainable_variables, tape=tape)

        ctc_decode_result1 = tf.keras.backend.ctc_decode(
            tf.cast(phone_output, tf.float32), input_length
        )[0][0]
        ctc_decode_result1 = tf.cast(
            tf.clip_by_value(ctc_decode_result1, 0, self.phone_num_classes),
            tf.int32,
        )

        ctc_decode_result2 = tf.keras.backend.ctc_decode(
            tf.cast(txt_output, tf.float32), new_input_length
        )[0][0]
        ctc_decode_result2 = tf.cast(
            tf.clip_by_value(ctc_decode_result2, 0, self.txt_num_classes),
            tf.int32,
        )

        ctc_decode_result3 = tf.keras.backend.ctc_decode(
            tf.cast(help_output, tf.float32), extra_phone_length
        )[0][0]
        ctc_decode_result3 = tf.cast(
            tf.clip_by_value(ctc_decode_result3, 0, self.txt_num_classes),
            tf.int32,
        )

        ctc_acc1 = self.ctc_acc(phone_labels, ctc_decode_result1)
        ctc_acc2 = self.ctc_acc(txt_label, ctc_decode_result2)
        ctc_acc3 = self.ctc_acc(extra_txts, ctc_decode_result3)
        return {
            "phone_loss": tf.reduce_mean(phone_ctc_loss) / self._distribution_strategy.num_replicas_in_sync,
            "txt_loss": tf.reduce_mean(txt_ctc_loss) / self._distribution_strategy.num_replicas_in_sync,
            "help_loss": tf.reduce_mean(help_ctc_loss) / self._distribution_strategy.num_replicas_in_sync,
            "phone_acc": tf.reduce_mean(ctc_acc1) / self._distribution_strategy.num_replicas_in_sync,
            "txt_acc": tf.reduce_mean(ctc_acc2) / self._distribution_strategy.num_replicas_in_sync,
            "help_acc": tf.reduce_mean(ctc_acc3) / self._distribution_strategy.num_replicas_in_sync,
            # 'final_acc':tf.reduce_mean(ctc_acc3)/self._distribution_strategy.num_replicas_in_sync,
        }

    def test_step(self, data):
        features, input_length, phone_labels, phone_label_length, txt_label, txt_label_lengths = data[0]

        max_T = tf.shape(phone_labels)[1]
        front_out = self.front(features, training=False)
        enc_output = self.encoder(front_out, training=False)
        phone_output, hidden_outputs = self.phone_picker(enc_output, training=False)
        picked_f, picked_c = self.feature_pick(hidden_outputs, phone_output, max_T)
        _, helper_out = self.helper.phone_call(phone_labels, training=False)
        picke_help = self.helper(picked_f, training=False)

        txt_output, _ = self.decoder(picke_help, training=False)
        help_output, _ = self.decoder(helper_out, training=False)

        T = tf.shape(picked_f)[1]
        new_input_length = tf.ones_like(input_length, tf.int32) * T

        phone_output = tf.nn.softmax(phone_output, -1)
        txt_output = tf.nn.softmax(txt_output, -1)
        help_output = tf.nn.softmax(help_output, -1)
        phone_ctc_loss = tf.keras.backend.ctc_batch_cost(
            tf.cast(phone_labels, tf.int32),
            tf.cast(phone_output, tf.float32),
            tf.cast(input_length[:, tf.newaxis], tf.int32),
            tf.cast(phone_label_length[:, tf.newaxis], tf.int32),
        )
        txt_ctc_loss = tf.keras.backend.ctc_batch_cost(
            tf.cast(txt_label, tf.int32),
            tf.cast(txt_output, tf.float32),
            tf.cast(new_input_length[:, tf.newaxis], tf.int32),
            tf.cast(txt_label_lengths[:, tf.newaxis], tf.int32),
        )

        help_ctc_loss = tf.keras.backend.ctc_batch_cost(
            tf.cast(txt_label, tf.int32),
            tf.cast(help_output, tf.float32),
            tf.cast(phone_label_length[:, tf.newaxis], tf.int32),
            tf.cast(txt_label_lengths[:, tf.newaxis], tf.int32),
        )

        ctc_decode_result1 = tf.keras.backend.ctc_decode(
            tf.cast(phone_output, tf.float32), input_length
        )[0][0]
        ctc_decode_result1 = tf.cast(
            tf.clip_by_value(ctc_decode_result1, 0, self.phone_num_classes),
            tf.int32,
        )

        ctc_decode_result2 = tf.keras.backend.ctc_decode(
            tf.cast(txt_output, tf.float32), new_input_length
        )[0][0]
        ctc_decode_result2 = tf.cast(
            tf.clip_by_value(ctc_decode_result2, 0, self.txt_num_classes),
            tf.int32,
        )

        ctc_decode_result3 = tf.keras.backend.ctc_decode(
            tf.cast(help_output, tf.float32), phone_label_length
        )[0][0]
        ctc_decode_result3 = tf.cast(
            tf.clip_by_value(ctc_decode_result3, 0, self.txt_num_classes),
            tf.int32,
        )

        ctc_acc1 = self.ctc_acc(phone_labels, ctc_decode_result1)
        ctc_acc2 = self.ctc_acc(txt_label, ctc_decode_result2)
        ctc_acc3 = self.ctc_acc(txt_label, ctc_decode_result3)
        return {
            "phone_loss": tf.reduce_mean(phone_ctc_loss) / self._distribution_strategy.num_replicas_in_sync,
            "txt_loss": tf.reduce_mean(txt_ctc_loss) / self._distribution_strategy.num_replicas_in_sync,
            "help_loss": tf.reduce_mean(help_ctc_loss) / self._distribution_strategy.num_replicas_in_sync,
            "phone_acc": tf.reduce_mean(ctc_acc1) / self._distribution_strategy.num_replicas_in_sync,
            "txt_acc": tf.reduce_mean(ctc_acc2) / self._distribution_strategy.num_replicas_in_sync,
            "help_acc": tf.reduce_mean(ctc_acc3) / self._distribution_strategy.num_replicas_in_sync,
            # 'final_acc':tf.reduce_mean(ctc_acc3)/self._distribution_strategy.num_replicas_in_sync,
        }
