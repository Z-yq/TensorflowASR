#encoding=utf-8
# import keras
import tensorflow as tf


class TFReflectionPad1d(tf.keras.layers.Layer):
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size, padding_type="REFLECT", **kwargs):
        """Initialize TFReflectionPad1d module.

        Args:
            padding_size (int)
            padding_type (str) ("CONSTANT", "REFLECT", or "SYMMETRIC". Default is "REFLECT")
        """
        super().__init__(**kwargs)
        self.padding_size = padding_size
        self.padding_type = padding_type

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Padded tensor (B, T + 2 * padding_size, C).
        """
        return tf.pad(x, [[0, 0], [self.padding_size, self.padding_size], [0, 0]], self.padding_type)
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        discriminator = []
        discriminator+=[tf.keras.layers.Conv1D(32,3,padding='same')]
        for i in range(1,5):
            discriminator+=[
                tf.keras.layers.Conv1D(32 * (2 ** i), 4, strides=2, padding='same'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv1D(32 * (2 ** i), 5, padding='same'),
            ]

        discriminator += [
            tf.keras.layers.Activation('sigmoid'),

        ]
        self.dis=tf.keras.Sequential(discriminator)
        self.conv=tf.keras.layers.Conv1D(128, 3, padding='same')
        self.final=tf.keras.layers.Dense(1, activation='sigmoid')

    def _build(self):
        self(tf.ones([1,1600,1]))


    def call(self,x):
        fea_out=self.dis(x)
        x=self.conv(fea_out)
        x=self.final(x)
        return fea_out,x

class TFResidualStack(tf.keras.layers.Layer):
    """Tensorflow ResidualStack module."""

    def __init__(self,
                 kernel_size=5,
                 filters=32,
                 dilation_rate=1,
                 use_bias=True,

                 **kwargs):
      
        super().__init__(**kwargs)
        self.blocks = [
            tf.keras.layers.LeakyReLU(),
            TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate),
            tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
             
            ),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv1D(filters=filters,
                                   kernel_size=1,
                                   use_bias=use_bias,)
                                  
        ]
        self.shortcut = tf.keras.layers.Conv1D(filters=filters,
                                               kernel_size=1,
                                               use_bias=use_bias,
                                        
                                               name='shortcut')


    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T, C).
        """
        _x = tf.identity(x)
        for layer in self.blocks:
            _x = layer(_x)
        shortcut = self.shortcut(x)
        return shortcut + _x



class WavePickModel(tf.keras.layers.Layer):
    def __init__(self,dout,hop_size):
        super().__init__()

        scales=self.get_scales(hop_size)
        layers=[]
        layers+=[
            tf.keras.layers.SeparableConv1D(filters=32, kernel_size=7, strides=scales[0], padding='same'),
            tf.keras.layers.LeakyReLU(),

                 ]
        for i in range(1,len(scales)):
            layers+=[
                tf.keras.layers.Conv1D(filters=min((32*(i+1)),dout), kernel_size=3, strides=scales[i],padding='same', kernel_regularizer=None, ),
                TFResidualStack(filters=min((32*(i+1)),dout)),
            ]
        layers+=[tf.keras.layers.Conv1D(filters=dout, kernel_size=7, strides=1, padding='same', kernel_regularizer=None, ),
               ]
        self.generator=tf.keras.Sequential(layers)

    # @tf.function(input_signature=[tf.TensorSpec([None, None, 1])])
    def call(self,x,training=True):
        return self.generator(x,training=training)

    def get_scales(self, num):
        scale = []
        while 1:
            for i in range(2, 100):
                if num % i == 0:
                    num = num // i
                    scale.append(i)
                    break
            if num == 1:
                break
        while len(scale) > 4:
            new_scale = scale[2:]
            new_scale.append(scale[0] * scale[1])
            scale = new_scale
            scale.sort()
        return scale[::-1]