import tensorflow as tf
class LayerNormLSTMCell(tf.keras.layers.LSTMCell):
    def __init__(
        self,
        units,
        activation = "tanh",
        recurrent_activation = "sigmoid",
        use_bias= True,
        kernel_initializer= "glorot_uniform",
        recurrent_initializer = "orthogonal",
        bias_initializer= "zeros",
        unit_forget_bias= True,
        kernel_regularizer= None,
        recurrent_regularizer = None,
        bias_regularizer= None,
        kernel_constraint = None,
        recurrent_constraint= None,
        bias_constraint= None,
        dropout = 0.0,
        recurrent_dropout = 0.0,
        norm_gamma_initializer = "ones",
        norm_beta_initializer = "zeros",
        norm_epsilon = 1e-3,
        **kwargs
    ):

        super().__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            **kwargs,
        )
        self.norm_gamma_initializer = tf.keras.initializers.get(norm_gamma_initializer)
        self.norm_beta_initializer = tf.keras.initializers.get(norm_beta_initializer)
        self.norm_epsilon = norm_epsilon
        self.kernel_norm = self._create_norm_layer("kernel_norm")
        self.recurrent_norm = self._create_norm_layer("recurrent_norm")
        self.state_norm = self._create_norm_layer("state_norm")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=4)
        if 0.0 < self.dropout < 1.0:
            inputs *= dp_mask[0]
        z = self.kernel_norm(tf.keras.backend.dot(inputs, self.kernel))

        if 0.0 < self.recurrent_dropout < 1.0:
            h_tm1 *= rec_dp_mask[0]
        z += self.recurrent_norm(tf.keras.backend.dot(h_tm1, self.recurrent_kernel))
        if self.use_bias:
            z = tf.keras.backend.bias_add(z, self.bias)

        z = tf.split(z, num_or_size_splits=4, axis=1)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)
        c = self.state_norm(c)
        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {
            "norm_gamma_initializer": tf.keras.initializers.serialize(
                self.norm_gamma_initializer
            ),
            "norm_beta_initializer": tf.keras.initializers.serialize(
                self.norm_beta_initializer
            ),
            "norm_epsilon": self.norm_epsilon,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _create_norm_layer(self, name):
        return tf.keras.layers.LayerNormalization(
            beta_initializer=self.norm_beta_initializer,
            gamma_initializer=self.norm_gamma_initializer,
            epsilon=self.norm_epsilon,
            name=name,
        )

