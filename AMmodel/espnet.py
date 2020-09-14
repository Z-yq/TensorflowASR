import tensorflow as tf
from AMmodel.layers.switchnorm import SwitchNormalization 
from utils.tools import shape_list
from AMmodel.conformer import ConformerBlock
from AMmodel.ctc_wrap import CtcModel
from AMmodel.transducer_wrap import Transducer
from AMmodel.las_wrap import LAS,LASConfig


class BN_PRelu(tf.keras.layers.Layer):
    """
    Does Batch Normalization followed by PReLU.
    """
    def __init__(self):
        super(BN_PRelu,self).__init__()
        self.sw=SwitchNormalization()
    def build(self, input_shape):
        self.alpha=self.add_weight(shape=[1],dtype=tf.float32,name='alpha')
    def call(self,x,training=False):
        x=self.sw(x,training=training)
        x=tf.maximum(0.0, x) + self.alpha * tf.minimum(0.0, x)
        return x
   





class ESP_alhpa(tf.keras.layers.Layer):
    """
    ESP-alpha module where alpha controls depth of network.
    Args:
        ip: Input
        n_out: number of output channels
    """
    def __init__(self,filter_size,kernel_size):
        super(ESP_alhpa,self).__init__()
        filter_size//=4
        self.chanle_projecter=tf.keras.layers.Conv2D(filter_size,kernel_size,padding='same')
        self.dilated_conv1 = tf.keras.layers.Conv2D(filter_size,kernel_size,padding='same',
                                   kernel_initializer=tf.random_uniform_initializer(seed=42),dilation_rate=(1, 1))
        self.dilated_conv2 = tf.keras.layers.Conv2D(filter_size,kernel_size,padding='same',
                                   kernel_initializer=tf.random_uniform_initializer(seed=42), dilation_rate=(2, 2))
        self.dilated_conv4 = tf.keras.layers.Conv2D(filter_size,kernel_size,padding='same',
                                   kernel_initializer=tf.random_uniform_initializer(seed=42), dilation_rate=(4, 4))
        self.dilated_conv8 = tf.keras.layers.Conv2D(filter_size,kernel_size,padding='same',
                                   kernel_initializer=tf.random_uniform_initializer(seed=42), dilation_rate=(8, 8))
        self.dilated_conv16 = tf.keras.layers.Conv2D(filter_size,kernel_size,padding='same',
                                   kernel_initializer=tf.random_uniform_initializer(seed=42), dilation_rate=(16, 16))
        
        self.sw=SwitchNormalization()
    def call(self,inputs,training=False):
        inputs_project=self.chanle_projecter(inputs)
        out1=self.dilated_conv1(inputs_project)
        add1=out1+inputs_project
        out2=self.dilated_conv2(add1)
        add2=out2+add1
        out3=self.dilated_conv4(add2)
        add3=out3+add2
        out4=self.dilated_conv8(add3)
        add4=out4+add3
        out5=self.dilated_conv16(add4)
        concat=tf.concat([out1,out2,out3,out4,out5],-1)
        concat=self.sw(concat,training=training)
        concat=tf.nn.leaky_relu(concat)
        return concat
class ESP_Block(tf.keras.layers.Layer):
    def __init__(self,num,filter_size,kernel_size,scale=2):
        super(ESP_Block,self).__init__()
        self.down=tf.keras.layers.Conv2D(filter_size,(4,4),strides=(scale,scale),padding='same')
        self.esp1=ESP_alhpa(filter_size,kernel_size)
        self.esp2=ESP_alhpa(filter_size,kernel_size)
        self.esps=[ESP_alhpa(filter_size,kernel_size) for i in range(num)]
      
        
    def call(self,inputs,training=False):
        down=self.down(inputs)
        down=self.esp1(down,training=training)
        x=down
        for layer in self.esps:
            plus=x
            x=layer(x,training=training)
            x+=plus
        # x=self.bnl(x)
        x=self.esp2(x,training=training)
        return x
        
        
class ESPNet(tf.keras.Model):
    def __init__(self,filter_size=128,
                        kernel_size=3,
                 block_1_num=4,
                 block_2_num=6,
                 dropout=0.1,
                 fc_factor=0.5, 
                 head_size=64, 
                 num_heads=4,
                 **kwargs
                        ):
        super(ESPNet,self).__init__()
        self.in_conv=tf.keras.layers.Conv2D(filter_size,kernel_size,name='in_conv')
        self.BN_Prelu1=BN_PRelu()
        self.espblock1=ESP_Block(block_1_num,filter_size,kernel_size,2)
        self.block1_projecter=tf.keras.layers.Conv2D(filter_size,kernel_size,padding='same',name='block1_projecter')
        self.espblock2=ESP_Block(block_2_num,filter_size,kernel_size,2)
        self.last_block=ConformerBlock(filter_size,dropout,fc_factor,head_size,num_heads,
                                       kernel_size)
        self.projecter=tf.keras.layers.Dense(filter_size)
        self.avg=tf.keras.layers.AveragePooling2D(padding='same')
    def call(self,inputs,training=False):
        x=self.in_conv(inputs)
        prelu=self.BN_Prelu1(x,training=training)
        avg_pl=self.avg(prelu)
        esp_1=self.espblock1(prelu,training=training)
        esp_1_out=self.block1_projecter(esp_1)
        esp_2_input=tf.concat([avg_pl,esp_1_out],-1)
        esp_2=self.espblock2(esp_2_input,training=training)
        b,w,h,c=shape_list(esp_2)
        esp_2=tf.reshape(esp_2,[b,w,h*c])
        out = self.projecter(esp_2)
        out=self.last_block(out,training=training)
        # print(out.shape)

        return out
        
class ESPNetCTC(CtcModel):
    def __init__(self,
                 model_config: dict,
                 num_classes: int,
                 name: str = "ESPNet",
                 speech_config=dict):
        super(ESPNetCTC, self).__init__(
            encoder= ESPNet(**model_config),
            num_classes=num_classes,
            name=f"{name}_ctc",
            speech_config=speech_config
        )
        self.time_reduction_factor = 4

class ESPNetLAS(LAS):
    def __init__(self,
                 config,
                 training,
                 name: str = "LAS",
                 enable_tflite_convertible=False,
                 speech_config=dict):
        config['LAS_decoder'].update({'encoder_dim': config['model_config']['filter_size']})
        decoder_config = LASConfig(**config['LAS_decoder'])

        super(ESPNetLAS, self).__init__(
            encoder= ESPNet(**config['model_config']),
            config=decoder_config, training=training,enable_tflite_convertible=enable_tflite_convertible,
        name=name,speech_config=speech_config)
        self.time_reduction_factor = 4

class ESPNetTransducer(Transducer):
    def __init__(self,
                 config,
                 name: str = "ESPNet",
                 speech_config=dict):

        super(ESPNetTransducer, self).__init__(
            encoder= ESPNet(**config['model_config']),
            vocabulary_size=config['Transducer_decoder']['vocabulary_size'],
            embed_dim=config['Transducer_decoder']['embed_dim'],
            embed_dropout=config['Transducer_decoder']['embed_dropout'],
            num_lstms=config['Transducer_decoder']['num_lstms'],
            lstm_units=config['Transducer_decoder']['lstm_units'],
            joint_dim=config['Transducer_decoder']['joint_dim'],
            name=name+'_transducer',
            speech_config=speech_config
        )
        self.time_reduction_factor = 4


    