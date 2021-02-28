import collections
import numpy as np

import tensorflow as tf

from tensorflow_addons.seq2seq import Sampler
from tensorflow_addons.seq2seq import BahdanauAttention

from tensorflow_addons.seq2seq import Decoder
from AMmodel.layers.decoder import dynamic_decode

from AMmodel.layers.LayerNormLstmCell import LayerNormLSTMCell


class AttentionConfig():
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
                 decoder_lstm_units=512,
                 attention_dim=768,
                 attention_filters=32,
                 attention_kernel=31,
                 encoder_dim=128,
                 startid=0,
                 ):
        self.embedding_hidden_size = embedding_hidden_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.n_prenet_layers = n_prenet_layers
        self.prenet_units = prenet_units
        self.prenet_activation = prenet_activation
        self.prenet_dropout_rate = prenet_dropout_rate
        self.n_lstm_decoder = n_lstm_decoder
        self.attention_dim = attention_dim
        self.attention_filters = attention_filters
        self.attention_kernel = attention_kernel
        self.n_classes = n_classes
        self.encoder_dim = encoder_dim
        self.startid = startid
        self.decoder_lstm_units = decoder_lstm_units


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


class AttentionSampler(Sampler):
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
        self.targets = targets
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


class LocationSensitiveAttention(BahdanauAttention):

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
        self.location_layer = tf.keras.layers.Conv1D(
            config.attention_dim, config.attention_kernel,padding='same', use_bias=False, name="location_layer"
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
        "context",
        "time",
        "state",
        "alignment_history",
        "max_alignments",
    ],
)

DecoderOutput = collections.namedtuple(
    "DecoderOutput", ("classes_output", "sample_id")
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

    def __init__(self,
                 config,
                 fc_layer,
                 embed_layer,
                 **kwargs):
        """Init variables."""
        super().__init__(**kwargs)

        self.attention_lstm = LayerNormLSTMCell(
            units=config.decoder_lstm_units, name="attention_lstm_cell"
        )
        if embed_layer is None:
            self.decoder_embedding = tf.keras.layers.Embedding(config.n_classes, config.embedding_hidden_size)
        else:
            self.decoder_embedding=embed_layer
        self.prenet = Prenet(config, name="prenet")

        self.attention_layer = LocationSensitiveAttention(
            config,
            memory=None,
            mask_encoder=True,
            memory_sequence_length=None,
            is_cumulate=True,
        )
        self.classes_projection = fc_layer

        self.config = config

    def set_alignment_size(self, alignment_size):
        self.alignment_size = alignment_size

    @property
    def output_size(self):
        """Return output (mel) size."""
        return self.config.n_classes

    @property
    def state_size(self):
        """Return hidden state size."""
        return DecoderCellState(
            attention_lstm_state=self.attention_lstm.state_size,
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

        initial_context = tf.zeros(
            shape=[batch_size, self.config.encoder_dim], dtype=tf.float32
        )
        initial_state = self.attention_layer.get_initial_state(
            batch_size, size=self.alignment_size
        )

        initial_alignment_history = tf.TensorArray(
                dtype=tf.float32, size=0, dynamic_size=True
            )
        return DecoderCellState(
            attention_lstm_state=initial_attention_lstm_cell_states,
            time=tf.zeros([], dtype=tf.int32),
            context=initial_context,
            state=initial_state,
            alignment_history=initial_alignment_history,
            max_alignments=tf.zeros([batch_size], dtype=tf.int32),
        )

    def call(self, inputs, states,training=False):
        """Call logic."""
        # print(inputs.shape)
        decoder_input = self.decoder_embedding(inputs)[:, 0, :]

        # 1. apply prenet for decoder_input.
        prenet_out = self.prenet(
            decoder_input, training=training
        )  # [batch_size, dim]

        # 2. concat prenet_out and prev context vector
        # then use it as input of attention lstm layer.
        attention_lstm_input = tf.concat([prenet_out, states.context], axis=-1)
        attention_lstm_output, next_attention_lstm_state = self.attention_lstm(
            attention_lstm_input, states.attention_lstm_state
        )

        # 3. compute context, alignment and cumulative alignment.
        prev_state = states.state

        prev_alignment_history = states.alignment_history
        prev_max_alignments = states.max_alignments
        context, alignments, state = self.attention_layer(
            [attention_lstm_output, prev_state, prev_max_alignments],
            training=training,
        )

        decoder_outputs = self.classes_projection(context[:, tf.newaxis, :])
        decoder_outputs = tf.squeeze(decoder_outputs, 1)

        alignment_history = prev_alignment_history.write(states.time, alignments)

        new_states = DecoderCellState(
            attention_lstm_state=next_attention_lstm_state,
            time=states.time + 1,
            context=context,
            state=state,
            alignment_history=alignment_history,
            max_alignments=tf.argmax(alignments, -1, output_type=tf.int32),
        )

        return decoder_outputs, new_states


class AttentionDecoder(Decoder):

    def __init__(self,
                 decoder_cell,
                 decoder_sampler,
                 output_layer=None,
                 ):
        """Initial variables."""
        self.cell = decoder_cell
        self.sampler = decoder_sampler
        self.output_layer = output_layer

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

            sample_id=self.sampler.sample_ids_shape  # tf.TensorShape([])
        )

    @property
    def output_dtype(self):
        return DecoderOutput(tf.float32, self.sampler.sample_ids_dtype)

    @property
    def batch_size(self):
        return self.sampler._batch_size

    def step(self, time, inputs, state, training=False):
        classes_outputs, cell_state = self.cell(
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
        )

        outputs = DecoderOutput(classes_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished)


class CTCAttention(tf.keras.Model):

    def __init__(self, encoder_dim, n_classes, fc_layer,embed_layer=None, **kwargs):
        super().__init__(self, **kwargs)
        self.att_config = AttentionConfig(n_classes, encoder_dim=encoder_dim)
        self.n_classes=n_classes
        self.decoder_cell = DecoderCell(self.att_config, fc_layer,embed_layer,name="decoder_cell")
        self.decoder = AttentionDecoder(
            self.decoder_cell,
            AttentionSampler(self.att_config)
        )
        self.maximum_iterations=1000
    def setup_window(self, win_front, win_back):
        """Call only for inference."""
        self.use_window_mask = True
        self.win_front = win_front
        self.win_back = win_back

    def setup_maximum_iterations(self, maximum_iterations):
        """Call only for inference."""
        self.maximum_iterations = maximum_iterations

    def call(self,
             encoder_hidden_states,
             input_lengths,
             targets,
             targets_lengths,
             training=False,
             ):
        targets = tf.expand_dims(targets, -1)

        batch_size = tf.shape(encoder_hidden_states)[0]
        alignment_size = tf.shape(encoder_hidden_states)[1]

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

        # run decode step.
        (
            (classes_prediction, _),
            final_decoder_state,
            _,
        ) = dynamic_decode(self.decoder,
                           maximum_iterations=self.maximum_iterations,
                           enable_tflite_convertible=False,
                           training=training)

        decoder_output = tf.reshape(
            classes_prediction, [batch_size, -1,  self.n_classes]
        )

        alignment_history = tf.transpose(
            final_decoder_state.alignment_history.stack(), [1, 2, 0]
        )

        return decoder_output, alignment_history
