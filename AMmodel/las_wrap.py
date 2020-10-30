import collections
import numpy as np

import tensorflow as tf

from tensorflow_addons.seq2seq import Sampler
from tensorflow_addons.seq2seq import BahdanauAttention

from tensorflow_addons.seq2seq import Decoder
from AMmodel.layers.decoder import dynamic_decode
from AMmodel.layers.time_frequency import Spectrogram,Melspectrogram
# from tensorflow_addons.seq2seq import dynamic_decode

class LASConfig():
    def __init__(self,
                 n_classes,
                 embedding_hidden_size=512,
                 initializer_range=0.02,
                 layer_norm_eps=1e-6,
                 n_prenet_layers=2,
                 prenet_units=256,
                 prenet_activation="mish",
                 prenet_dropout_rate=0.5,
                 n_lstm_decoder=1,
                 decoder_lstm_units=1024,
                 attention_dim=128,
                 attention_filters=32,
                 attention_kernel=31,
                 encoder_dim=128,
                 startid=1,
                 ):

        self.embedding_hidden_size = embedding_hidden_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.n_prenet_layers = n_prenet_layers
        self.prenet_units = prenet_units
        self.prenet_activation = prenet_activation
        self.prenet_dropout_rate = prenet_dropout_rate
        self.n_lstm_decoder = n_lstm_decoder
        self.decoder_lstm_units = decoder_lstm_units
        self.attention_dim = attention_dim
        self.attention_filters = attention_filters
        self.attention_kernel = attention_kernel
        self.n_classes = n_classes
        self.encoder_dim=encoder_dim
        self.startid=startid




def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
    Args:
        initializer_range: float, initializer range for stddev.
    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def gelu(x):
    """Gaussian Error Linear unit."""
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


def gelu_new(x):
    """Smoother gaussian Error Linear Unit."""
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    """Swish activation function."""
    return x * tf.sigmoid(x)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


ACT2FN = {
    "identity": tf.keras.layers.Activation("linear"),
    "tanh": tf.keras.layers.Activation("tanh"),
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
    "gelu_new": tf.keras.layers.Activation(gelu_new),
    "mish": tf.keras.layers.Activation(mish),
}


class TrainingSampler(Sampler):
    """Training sampler for Seq2Seq training."""

    def __init__(
            self, config,
    ):
        super().__init__()
        self.config = config

        self._reduction_factor = 1

    def setup_target(self, targets, targets_lengths):
        """Setup ground-truth mel outputs for decoder."""
        self.targets_lengths = targets_lengths
        self.set_batch_size(tf.shape(targets)[0])
        self.targets = targets[
                       :, self._reduction_factor - 1:: self._reduction_factor, :
                       ]
        self.max_lengths = tf.tile([tf.shape(self.targets)[1]], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    @property
    def reduction_factor(self):
        return self._reduction_factor

    def initialize(self):
        """Return (Finished, next_inputs)."""
        return (
            tf.tile([False], [self._batch_size]),
            tf.tile([[self.config.startid]], [self._batch_size, self._reduction_factor]),
        )

    def sample(self, time, outputs, state):
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, **kwargs):
        finished = time + 1 >= self.max_lengths
        next_inputs = self.targets[:, time, :]

        next_state = state
        return (finished, next_inputs, next_state)

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size


class TestingSampler(TrainingSampler):
    """Testing sampler for Seq2Seq training."""

    def __init__(
            self, config,
    ):
        super().__init__(config)

    def next_inputs(self, time, outputs, state, sample_ids, **kwargs):
        stop_token_prediction = kwargs.get("stop_token_prediction")
        stop_token_prediction = tf.nn.sigmoid(stop_token_prediction)
        finished = tf.cast(tf.round(stop_token_prediction), tf.bool)
        finished = tf.reduce_all(finished)
        next_inputs = tf.expand_dims(tf.argmax(outputs[:, -self.config.n_classes:], -1,tf.int32), axis=-1, name='predict')
        next_state = state
        return (finished, next_inputs, next_state)


class LocationSensitiveAttention(BahdanauAttention):
    """Tacotron-2 Location Sensitive Attention module."""

    def __init__(
            self,
            config,
            memory,
            mask_encoder=True,
            memory_sequence_length=None,
            is_cumulate=True,
    ):
        """Init variables."""
        memory_length = memory_sequence_length if (mask_encoder is True) else None
        super().__init__(
            units=config.attention_dim,
            memory=memory,
            memory_sequence_length=memory_length,
            probability_fn="softmax",
            name="LocationSensitiveAttention",
        )
        self.location_convolution = tf.keras.layers.Conv1D(
            filters=config.attention_filters,
            kernel_size=config.attention_kernel,
            padding="same",
            use_bias=False,
            name="location_conv",
        )
        self.location_layer = tf.keras.layers.Dense(
            units=config.attention_dim, use_bias=False, name="location_layer"
        )

        self.v = tf.keras.layers.Dense(1, use_bias=True, name="scores_attention")
        self.config = config
        self.is_cumulate = is_cumulate
        self.use_window = False

    def setup_window(self, win_front=2, win_back=4):
        self.win_front = tf.constant(win_front, tf.int32)
        self.win_back = tf.constant(win_back, tf.int32)

        self._indices = tf.expand_dims(tf.range(tf.shape(self.keys)[1]), 0)
        self._indices = tf.tile(
            self._indices, [tf.shape(self.keys)[0], 1]
        )  # [batch_size, max_time]

        self.use_window = True

    def _compute_window_mask(self, max_alignments):
        """Compute window mask for inference.
        Args:
            max_alignments (int): [batch_size]
        """
        expanded_max_alignments = tf.expand_dims(max_alignments, 1)  # [batch_size, 1]
        low = expanded_max_alignments - self.win_front
        high = expanded_max_alignments + self.win_back
        mlow = tf.cast((self._indices < low), tf.float32)
        mhigh = tf.cast((self._indices > high), tf.float32)
        mask = mlow + mhigh
        return mask  # [batch_size, max_length]

    def __call__(self, inputs, training=False):
        query, state, prev_max_alignments = inputs

        processed_query = self.query_layer(query) if self.query_layer else query
        processed_query = tf.expand_dims(processed_query, 1)

        expanded_alignments = tf.expand_dims(state, axis=2)
        f = self.location_convolution(expanded_alignments)
        processed_location_features = self.location_layer(f)

        energy = self._location_sensitive_score(
            processed_query, processed_location_features, self.keys
        )

        # mask energy on inference steps.
        if self.use_window is True:
            window_mask = self._compute_window_mask(prev_max_alignments)
            energy = energy + window_mask * -1e20

        alignments = self.probability_fn(energy, state)

        if self.is_cumulate:
            state = alignments + state
        else:
            state = alignments

        expanded_alignments = tf.expand_dims(alignments, 2)
        context = tf.reduce_sum(expanded_alignments * self.values, 1)

        return context, alignments, state

    def _location_sensitive_score(self, W_query, W_fil, W_keys):
        """Calculate location sensitive energy."""
        return tf.squeeze(self.v(tf.nn.tanh(W_keys + W_query + W_fil)), -1)

    def get_initial_state(self, batch_size, size):
        """Get initial alignments."""
        return tf.zeros(shape=[batch_size, size], dtype=tf.float32)

    def get_initial_context(self, batch_size):
        """Get initial attention."""
        return tf.zeros(
            shape=[batch_size, self.config.encoder_lstm_units * 2], dtype=tf.float32
        )


DecoderCellState = collections.namedtuple(
    "DecoderCellState",
    [
        "attention_lstm_state",
        "decoder_lstms_state",
        "context",
        "time",
        "state",
        "alignment_history",
        "max_alignments",
    ],
)

DecoderOutput = collections.namedtuple(
    "DecoderOutput", ("classes_output", "token_output", "sample_id")
)

class Prenet(tf.keras.layers.Layer):
    """Tacotron-2 prenet."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.prenet_dense = [
            tf.keras.layers.Dense(
                units=config.prenet_units,
                activation=ACT2FN[config.prenet_activation],
                name="dense_._{}".format(i),
            )
            for i in range(config.n_prenet_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(
            rate=config.prenet_dropout_rate, name="dropout"
        )

    def call(self, inputs, training=False):
        """Call logic."""
        outputs = inputs
        for layer in self.prenet_dense:
            outputs = layer(outputs)
            outputs = self.dropout(outputs, training=True)
        return outputs

class DecoderCell(tf.keras.layers.AbstractRNNCell):
    """Tacotron-2 custom decoder cell."""

    def __init__(self,
                 config,
                 training,
                 enable_tflite_convertible=True,
                 **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.training = training
        self.enable_tflite_convertible = enable_tflite_convertible
        self.attention_lstm = tf.keras.layers.LSTMCell(
            units=config.decoder_lstm_units, name="attention_lstm_cell"
        )
        self.decoder_embedding = tf.keras.layers.Embedding(config.n_classes, config.embedding_hidden_size)
        lstm_cells = []
        for i in range(config.n_lstm_decoder):
            lstm_cell = tf.keras.layers.LSTMCell(
                units=config.decoder_lstm_units, name="lstm_cell_._{}".format(i)
            )
            lstm_cells.append(lstm_cell)
        self.decoder_lstms = tf.keras.layers.StackedRNNCells(
            lstm_cells, name="decoder_lstms"
        )
        self.prenet = Prenet(config, name="prenet")
        # define attention layer.

        # create location-sensitive attention.

        self.attention_layer = LocationSensitiveAttention(
            config,
            memory=None,
            mask_encoder=True,
            memory_sequence_length=None,
            is_cumulate=True,
        )
        self.classes_projection = tf.keras.layers.Dense(
            units=config.n_classes, name="classes_projection"
        )
        self.stop_projection = tf.keras.layers.Dense(
            units=1, name="stop_projection"
        )

        self.config = config

    def set_alignment_size(self, alignment_size):
        self.alignment_size = alignment_size

    @property
    def output_size(self):
        """Return output (mel) size."""
        return self.classes_projection.units

    @property
    def state_size(self):
        """Return hidden state size."""
        return DecoderCellState(
            attention_lstm_state=self.attention_lstm.state_size,
            decoder_lstms_state=self.decoder_lstms.state_size,
            time=tf.TensorShape([]),
            attention=self.config.attention_dim,
            state=self.alignment_size,
            alignment_history=(),
            max_alignments=tf.TensorShape([1]),
        )

    def get_initial_state(self, batch_size):
        """Get initial states."""
        initial_attention_lstm_cell_states = self.attention_lstm.get_initial_state(
            None, batch_size, dtype=tf.float32
        )
        initial_decoder_lstms_cell_states = self.decoder_lstms.get_initial_state(
            None, batch_size, dtype=tf.float32
        )
        initial_context = tf.zeros(
            shape=[batch_size, self.config.encoder_dim], dtype=tf.float32
        )
        initial_state = self.attention_layer.get_initial_state(
            batch_size, size=self.alignment_size
        )
        if self.enable_tflite_convertible:
            initial_alignment_history = ()
        else:
            initial_alignment_history = tf.TensorArray(
                dtype=tf.float32, size=0, dynamic_size=True
            )
        return DecoderCellState(
            attention_lstm_state=initial_attention_lstm_cell_states,
            decoder_lstms_state=initial_decoder_lstms_cell_states,
            time=tf.zeros([], dtype=tf.int32),
            context=initial_context,
            state=initial_state,
            alignment_history=initial_alignment_history,
            max_alignments=tf.zeros([batch_size], dtype=tf.int32),
        )

    def call(self, inputs, states):
        """Call logic."""
        # tf.print(inputs.shape)
        decoder_input = self.decoder_embedding(inputs)[:,0,:]

        # 1. apply prenet for decoder_input.
        prenet_out = self.prenet(
            decoder_input, training=self.training
        )  # [batch_size, dim]

        # 2. concat prenet_out and prev context vector
        # then use it as input of attention lstm layer.
        attention_lstm_input = tf.concat([prenet_out, states.context], axis=-1)
        attention_lstm_output, next_attention_lstm_state = self.attention_lstm(
            attention_lstm_input, states.attention_lstm_state
        )

        # 3. compute context, alignment and cumulative alignment.
        prev_state = states.state
        if not self.enable_tflite_convertible:
            prev_alignment_history = states.alignment_history
        prev_max_alignments = states.max_alignments
        context, alignments, state = self.attention_layer(
            [attention_lstm_output, prev_state, prev_max_alignments],
            training=self.training,
        )

        # 4. run decoder lstm(s)
        decoder_lstms_input = tf.concat([attention_lstm_output, context], axis=-1)
        decoder_lstms_output, next_decoder_lstms_state = self.decoder_lstms(
            decoder_lstms_input, states.decoder_lstms_state
        )

        # 5. compute frame feature and stop token.
        projection_inputs = tf.concat([decoder_lstms_output, context], axis=-1)
        decoder_outputs = self.classes_projection(projection_inputs)

        stop_inputs = tf.concat([decoder_lstms_output, decoder_outputs], axis=-1)
        stop_tokens = self.stop_projection(stop_inputs)

        # 6. save alignment history to visualize.
        if self.enable_tflite_convertible:
            alignment_history = ()
        else:
            alignment_history = prev_alignment_history.write(states.time,
                                                             alignments)

        # 7. return new states.
        new_states = DecoderCellState(
            attention_lstm_state=next_attention_lstm_state,
            decoder_lstms_state=next_decoder_lstms_state,
            time=states.time + 1,
            context=context,
            state=state,
            alignment_history=alignment_history,
            max_alignments=tf.argmax(alignments, -1, output_type=tf.int32),
        )

        return (decoder_outputs, stop_tokens), new_states


class LASDecoder(Decoder):
    """LAS Decoder."""

    def __init__(self,
                 decoder_cell,
                 decoder_sampler,

                 output_layer=None,
                 enable_tflite_convertible=False):
        """Initial variables."""
        self.cell = decoder_cell
        self.sampler = decoder_sampler
        self.output_layer = output_layer
        self.enable_tflite_convertible = enable_tflite_convertible

    def setup_decoder_init_state(self, decoder_init_state):
        self.initial_state = decoder_init_state

    def initialize(self, **kwargs):
        return self.sampler.initialize() + (self.initial_state,)

    @property
    def output_size(self):
        return DecoderOutput(
            classes_output=tf.nest.map_structure(
                lambda shape: tf.TensorShape(shape), self.cell.output_size
            ),
            token_output=tf.TensorShape(self.sampler.reduction_factor),
            sample_id=tf.TensorShape([1]) \
                if self.enable_tflite_convertible \
                else self.sampler.sample_ids_shape  # tf.TensorShape([])
        )

    @property
    def output_dtype(self):
        return DecoderOutput(tf.float32, tf.float32, self.sampler.sample_ids_dtype)

    @property
    def batch_size(self):
        return self.sampler._batch_size

    def step(self, time, inputs, state, training=False):
        (classes_outputs, stop_tokens), cell_state = self.cell(
            inputs, state, training=training
        )
        if self.output_layer is not None:
            classes_outputs = self.output_layer(classes_outputs)
        sample_ids = self.sampler.sample(
            time=time, outputs=classes_outputs, state=cell_state
        )
        (finished, next_inputs, next_state) = self.sampler.next_inputs(
            time=time,
            outputs=classes_outputs,
            state=cell_state,
            sample_ids=sample_ids,
            stop_token_prediction=stop_tokens,
        )

        outputs = DecoderOutput(classes_outputs, stop_tokens, sample_ids)
        return (outputs, next_state, next_inputs, finished)


class LAS(tf.keras.Model):

    def __init__(self, encoder, config, training, enable_tflite_convertible=False,speech_config=dict, **kwargs):
        super().__init__(self, **kwargs)
        self.encoder = encoder
        self.decoder_cell = DecoderCell(
            config, training=training, name="decoder_cell",
            enable_tflite_convertible=enable_tflite_convertible
        )
        self.decoder = LASDecoder(
            self.decoder_cell,
            TrainingSampler(config) if training is True else TestingSampler(config),
            enable_tflite_convertible=enable_tflite_convertible
        )
        self.config = config
        self.speech_config = speech_config
        self.mel_layer = None
        if speech_config['use_mel_layer']:
            if speech_config['mel_layer_type'] == 'Melspectrogram':
                self.mel_layer = Melspectrogram(sr=speech_config['sample_rate'],
                                                n_mels=speech_config['num_feature_bins'],
                                                n_hop=int(speech_config['stride_ms'] * speech_config['sample_rate']//1000),
                                                n_dft=1024,
                                                trainable_fb=speech_config['trainable_kernel']
                                                )
            else:
                self.mel_layer = Spectrogram(
                                             n_hop=int(speech_config['stride_ms'] * speech_config['sample_rate']//1000),
                                             n_dft=1024,
                                             trainable_kernel=speech_config['trainable_kernel']
                                             )


        self.use_window_mask = False
        self.maximum_iterations = 1000 if training else 50
        self.enable_tflite_convertible = enable_tflite_convertible

    def setup_window(self, win_front, win_back):
        """Call only for inference."""
        self.use_window_mask = True
        self.win_front = win_front
        self.win_back = win_back

    def setup_maximum_iterations(self, maximum_iterations):
        """Call only for inference."""
        self.maximum_iterations = maximum_iterations

    def _build(self, shape, training):

        batch=shape[0]
        inputs = np.random.normal(size=shape).astype(np.float32)
        if self.mel_layer is not None:
            input_lengths = np.array([shape[1] // 4//self.mel_layer.n_hop] * batch, 'int32')
        else:
            input_lengths = np.array([shape[1]//4]*batch,'int32')

        if training:
            targets = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]]*batch)
            targets = targets[:, :, np.newaxis]
            targets_lengths = np.array([9]*batch)
            self([inputs,input_lengths],
                 targets,targets_lengths)
        else:
            self(
                [inputs,
                input_lengths],

                training=training,
            )
    def add_featurizers(self,
                        text_featurizer):
        """
        Function to add featurizer to model to convert to end2end tflite
        Args:
            text_featurizer: TextFeaturizer instance
            scorer: external language model scorer
        """

        self.text_featurizer = text_featurizer
    # @tf.function(experimental_relax_shapes=True)
    def call(
            self,
            inputs,
            targets=None,
            targets_lengths=None,
            use_window_mask=False,
            win_front=2,
            win_back=3,
            training=False,
    ):
        """Call logic."""
        # Encoder Step.
        # input_lengths=tf.squeeze(input_lengths,-1)
        inputs, input_lengths=inputs
        if self.mel_layer is not None:
            inputs=self.mel_layer(inputs)
        
        encoder_hidden_states = self.encoder(
            inputs, training=training
        )
        batch_size = tf.shape(encoder_hidden_states)[0]
        alignment_size = tf.shape(encoder_hidden_states)[1]

        # Setup some initial placeholders for decoder step. Include:
        # 1. mel_outputs, mel_lengths for teacher forcing mode.
        # 2. alignment_size for attention size.
        # 3. initial state for decoder cell.
        # 4. memory (encoder hidden state) for attention mechanism.
        if targets is not None:
            self.decoder.sampler.setup_target(targets=targets, targets_lengths=targets_lengths)
        self.decoder.sampler.set_batch_size(batch_size)
        self.decoder.cell.set_alignment_size(alignment_size)

        self.decoder.setup_decoder_init_state(
            self.decoder.cell.get_initial_state(batch_size)
        )
        self.decoder.cell.attention_layer.setup_memory(
            memory=encoder_hidden_states,
            memory_sequence_length=input_lengths,  # use for mask attention.
        )
        if use_window_mask:
            self.decoder.cell.attention_layer.setup_window(
                win_front=win_front, win_back=win_back
            )

        # run decode step.
        (
            (classes_prediction, stop_token_prediction, _),
            final_decoder_state,
            _,
        ) = dynamic_decode(self.decoder,
                           maximum_iterations=self.maximum_iterations,
                           enable_tflite_convertible=self.enable_tflite_convertible)

        decoder_output = tf.reshape(
            classes_prediction, [batch_size, -1, self.config.n_classes]
        )
        stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

        if self.enable_tflite_convertible:
            mask = tf.math.not_equal(
                tf.cast(tf.reduce_sum(tf.abs(decoder_output), axis=-1),
                        dtype=tf.int32),
                0)
            decoder_output = tf.expand_dims(
                tf.boolean_mask(decoder_output, mask), axis=0)
            alignment_history = ()
        else:
            alignment_history = tf.transpose(
                final_decoder_state.alignment_history.stack(), [1, 2, 0]
            )

        return decoder_output, stop_token_prediction, alignment_history
    def return_pb_function(self,shape):
        @tf.function(
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape, dtype=tf.float32),
                tf.TensorSpec([None, 1], dtype=tf.int32),
            ],
        )
        def inference( inputs, input_lengths):
            """Call logic."""

            # Encoder Step.
            input_lengths=tf.squeeze(input_lengths,-1)

            if self.mel_layer is not None:
                inputs=self.mel_layer(inputs)
            encoder_hidden_states = self.encoder.call(
                inputs, training=False
            )
            batch_size = tf.shape(encoder_hidden_states)[0]
            alignment_size = tf.shape(encoder_hidden_states)[1]

            # Setup some initial placeholders for decoder step. Include:
            # 1. batch_size for inference.
            # 2. alignment_size for attention size.
            # 3. initial state for decoder cell.
            # 4. memory (encoder hidden state) for attention mechanism.
            # 5. window front/back to solve long sentence synthesize problems. (call after setup memory.)
            self.decoder.sampler.set_batch_size(batch_size)
            self.decoder.cell.set_alignment_size(alignment_size)
            # self.setup_maximum_iterations(alignment_size)
            self.decoder.setup_decoder_init_state(
                self.decoder.cell.get_initial_state(batch_size)
            )
            self.decoder.cell.attention_layer.setup_memory(
                memory=encoder_hidden_states,
                memory_sequence_length=input_lengths,  # use for mask attention.
            )
            if self.use_window_mask:
                self.decoder.cell.attention_layer.setup_window(
                    win_front=self.win_front, win_back=self.win_back
                )

            (
                (classes_prediction, stop_token_prediction, _),
                final_decoder_state,
                _,
            ) = dynamic_decode(self.decoder, maximum_iterations=self.maximum_iterations)

            decoder_output = tf.reshape(
                classes_prediction, [batch_size, -1, self.config.n_classes]
            )
            stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

            alignment_history = tf.transpose(
                final_decoder_state.alignment_history.stack(), [1, 2, 0]
            )
            decoder_output=tf.argmax(decoder_output,-1)
            return [decoder_output]
        self.recognize_pb=inference


