
import tensorflow as tf
from AMmodel.transducer_wrap_cfm import Transducer
from AMmodel.ctc_wrap_cfm import CtcModel
from AMmodel.las_wrap_cfm import LAS,LASConfig
from AMmodel.conformer_blocks import ConformerEncoder

class ConformerTransducer(Transducer):
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
        super(ConformerTransducer, self).__init__(
            encoder=ConformerEncoder(
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
class ConformerCTC(CtcModel):
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
        super(ConformerCTC, self).__init__(
            encoder=ConformerEncoder(
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
class ConformerLAS(LAS):
    def __init__(self,
                 config,
                 training=True,
                 enable_tflite_convertible=False,
                 speech_config=dict,
                 ):
        config['LAS_decoder'].update({'encoder_dim':config['dmodel']})
        decoder_config=LASConfig(**config['LAS_decoder'])

        super(ConformerLAS,self).__init__(
            encoder=ConformerEncoder(
                dmodel=config['dmodel'],
                reduction_factor=config['reduction_factor'],
                num_blocks=config['num_blocks'],
                head_size=config['head_size'],
                num_heads=config['num_heads'],
                kernel_size=config['kernel_size'],
                fc_factor=config['fc_factor'],
                dropout=config['dropout'],
                name=config['name'],
                add_wav_info=speech_config['add_wav_info'],
                hop_size=int(speech_config['stride_ms'] * speech_config['sample_rate'] // 1000)*config['reduction_factor'],
            ),config=decoder_config,training=training,enable_tflite_convertible=enable_tflite_convertible,
        speech_config=speech_config
        )
        self.time_reduction_factor = config['reduction_factor']

