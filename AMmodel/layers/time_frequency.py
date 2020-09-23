import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from AMmodel.layers import backend, backend_keras
import tensorflow as tf

class Spectrogram(Layer):
    def __init__(self, n_dft=512, n_hop=None, padding='same',
                 power_spectrogram=2.0, return_decibel_spectrogram=True,
                 trainable_kernel=False, image_data_format='channels_last', **kwargs):
        assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
            ('n_dft should be > 1 and power of 2, but n_dft == %d' % n_dft)
        assert isinstance(trainable_kernel, bool)
        assert isinstance(return_decibel_spectrogram, bool)
        assert padding in ('same', 'valid')
        if n_hop is None:
            n_hop = n_dft // 2

        assert image_data_format in ('default', 'channels_first', 'channels_last')
        if image_data_format == 'default':
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format

        self.n_dft = n_dft
        assert n_dft % 2 == 0
        self.n_filter = n_dft // 2 + 1
        self.trainable_kernel = trainable_kernel
        self.n_hop = n_hop
        self.padding = padding
        self.power_spectrogram = float(power_spectrogram)
        self.return_decibel_spectrogram = return_decibel_spectrogram
        super(Spectrogram, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n_ch = input_shape[2]
        self.len_src = input_shape[1]
        self.is_mono = (self.n_ch == 1)
        if self.image_data_format == 'channels_first':
            self.ch_axis_idx = 1
        else:
            self.ch_axis_idx = 3
        if self.len_src is not None:
            assert self.len_src >= self.n_dft, 'Hey! The input is too short!'

        self.n_frame = conv_output_length(self.len_src,
                                          self.n_dft,
                                          self.padding,
                                          self.n_hop)

        dft_real_kernels, dft_imag_kernels = backend.get_stft_kernels(self.n_dft)
        self.dft_real_kernels = K.variable(dft_real_kernels, dtype=K.floatx(), name="real_kernels")
        self.dft_imag_kernels = K.variable(dft_imag_kernels, dtype=K.floatx(), name="imag_kernels")
        # kernels shapes: (filter_length, 1, input_dim, nb_filter)?
        if self.trainable_kernel:
            self.trainable_weights.append(self.dft_real_kernels)
            self.trainable_weights.append(self.dft_imag_kernels)
        else:
            self.non_trainable_weights.append(self.dft_real_kernels)
            self.non_trainable_weights.append(self.dft_imag_kernels)

        super(Spectrogram, self).build(input_shape)
        # self.built = True

    def compute_output_shape(self, input_shape):
        if self.image_data_format == 'channels_first':
            return input_shape[0], self.n_ch,  self.n_frame,self.n_filter
        else:
            return input_shape[0],  self.n_frame, self.n_filter,self.n_ch

    def call(self, x):
        x=tf.transpose(x,[0,2,1])
        output = self._spectrogram_mono(x[:, 0:1, :])
        if self.is_mono is False:
            for ch_idx in range(1, self.n_ch):
                output = K.concatenate((output,
                                        self._spectrogram_mono(x[:, ch_idx:ch_idx + 1, :])),
                                       axis=self.ch_axis_idx)
        if self.power_spectrogram != 2.0:
            output = K.pow(K.sqrt(output), self.power_spectrogram)
        if self.return_decibel_spectrogram:
            output = backend_keras.amplitude_to_decibel(output)

        return output

    def get_config(self):
        config = {'n_dft': self.n_dft,
                  'n_hop': self.n_hop,
                  'padding': self.padding,
                  'power_spectrogram': self.power_spectrogram,
                  'return_decibel_spectrogram': self.return_decibel_spectrogram,
                  'trainable_kernel': self.trainable_kernel,
                  'image_data_format': self.image_data_format}
        base_config = super(Spectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _spectrogram_mono(self, x):
        '''x.shape : (None, 1, len_src),
        returns 2D batch of a mono power-spectrogram'''
        x = K.permute_dimensions(x, [0, 2, 1])
        x = K.expand_dims(x, 3)  # add a dummy dimension (channel axis)
        subsample = (self.n_hop, 1)
        output_real = K.conv2d(x, self.dft_real_kernels,
                               strides=subsample,
                               padding=self.padding,
                               data_format='channels_last')
        output_imag = K.conv2d(x, self.dft_imag_kernels,
                               strides=subsample,
                               padding=self.padding,
                               data_format='channels_last')
        output = output_real ** 2 + output_imag ** 2
        # now shape is (batch_sample, n_frame, 1, freq)
        if self.image_data_format == 'channels_last':
            output = K.permute_dimensions(output, [0, 1,3, 2])
        else:
            output = K.permute_dimensions(output, [0, 2, 1, 3])
        return output


class Melspectrogram(Spectrogram):
    def __init__(self,
                 sr=16000, n_mels=128, fmin=0.0, fmax=None,
                 power_melgram=2.0, return_decibel_melgram=True,
                 trainable_fb=False, htk=False, norm=1, **kwargs):

        super(Melspectrogram, self).__init__(**kwargs)
        assert sr > 0
        assert fmin >= 0.0
        if fmax is None:
            fmax = float(sr) / 2
        assert fmax > fmin
        assert isinstance(return_decibel_melgram, bool)
        if 'power_spectrogram' in kwargs:
            assert kwargs['power_spectrogram'] == 2.0, \
                'In Melspectrogram, power_spectrogram should be set as 2.0.'

        self.sr = int(sr)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.return_decibel_melgram = return_decibel_melgram
        self.trainable_fb = trainable_fb
        self.power_melgram = power_melgram
        self.htk = htk
        self.norm = norm

    def build(self, input_shape):
        super(Melspectrogram, self).build(input_shape)
        self.built = False
        # compute freq2mel matrix --> 
        mel_basis = backend.mel(self.sr, self.n_dft, self.n_mels, self.fmin, self.fmax,
                                self.htk, self.norm)  # (128, 1025) (mel_bin, n_freq)
        mel_basis = np.transpose(mel_basis)

        self.freq2mel = K.variable(mel_basis, dtype=K.floatx())
        if self.trainable_fb:
            self.trainable_weights.append(self.freq2mel)
        else:
            self.non_trainable_weights.append(self.freq2mel)
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.image_data_format == 'channels_first':
            return input_shape[0], self.n_ch, self.n_mels, self.n_frame
        else:
            return input_shape[0], self.n_mels, self.n_frame, self.n_ch

    def call(self, x):
        power_spectrogram = super(Melspectrogram, self).call(x)

        if self.image_data_format == 'channels_first':
            pass
        else:
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 3, 1, 2])
        output = K.dot(power_spectrogram, self.freq2mel)
        if self.image_data_format == 'channels_first':
           pass
        else:
            output = K.permute_dimensions(output, [0, 2, 3, 1])
        if self.power_melgram != 2.0:
            output = K.pow(K.sqrt(output), self.power_melgram)
        # if self.return_decibel_melgram:
        #     output = backend_keras.amplitude_to_decibel(output)
        return output

    def get_config(self):
        config = {'sr': self.sr,
                  'n_mels': self.n_mels,
                  'fmin': self.fmin,
                  'fmax': self.fmax,
                  'trainable_fb': self.trainable_fb,
                  'power_melgram': self.power_melgram,
                  'return_decibel_melgram': self.return_decibel_melgram,
                  'htk': self.htk,
                  'norm': self.norm}
        base_config = super(Melspectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    """Determines output length of a convolution given input length.
    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        stride: integer.
        dilation: dilation rate, integer.
    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = (filter_size - 1) * dilation + 1
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride