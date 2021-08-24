
import tensorflow as tf
from AMmodel.stream_transducer_wrap import Transducer
from AMmodel.stream_ctc_wrap import CtcModel
from AMmodel.conformer_blocks import ConformerEncoder

class StreamingConformerEncoder(ConformerEncoder):
    def add_chunk_size(self,chunk_size,mel_size,hop_size):
        self.chunk_size=chunk_size
        self.mel_size=mel_size
        self.mel_length=self.chunk_size//hop_size if self.chunk_size%hop_size==0 else self.chunk_size//hop_size+1
        print(self.chunk_size,self.mel_size,self.mel_length)

    def call(self, inputs, training=False, **kwargs):

        if self.add_wav_info:
            mel_inputs, wav_inputs = inputs
            B = tf.shape(mel_inputs)[0]
            mel_inputs = tf.reshape(mel_inputs, [-1, self.mel_length, self.mel_size, 1])
            wav_inputs = tf.reshape(wav_inputs, [-1, self.chunk_size, 1])
            mel_outputs = self.conv_subsampling(mel_inputs, training=training)
            wav_outputs = self.wav_layer(wav_inputs, training=training)
            outputs = mel_outputs + wav_outputs
        else:
            B=tf.shape(inputs)[0]
            inputs = tf.reshape(inputs, [-1, self.mel_length, self.mel_size, 1])
            outputs = self.conv_subsampling(inputs, training=training)

        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, training=training)
        outputs = tf.reshape(outputs, [B, -1, self.dmodel])
        return outputs
    def inference(self, inputs, training=False, **kwargs):
        if self.add_wav_info:
            mel_inputs, wav_inputs = inputs
            mel_outputs = self.conv_subsampling(mel_inputs, training=training)
            wav_outputs = self.wav_layer(wav_inputs, training=training)
            outputs = mel_outputs+wav_outputs
        else:
            outputs = self.conv_subsampling(inputs, training=training)

        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, training=training)
        return outputs

class StreamingConformerTransducer(Transducer):
    def __init__(self,
                 dmodel: int,
                 reduction_factor: int,
                 vocabulary_size: int,
                 num_blocks: int = 8,
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




