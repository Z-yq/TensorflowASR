
import numpy as np
import os
from utils.text_featurizers import TextFeaturizer
from utils.speech_featurizers import SpeechFeaturizer

import logging

class AM():
    def __init__(self,config):
        self.config = config
        self.update_model_type()
        self.speech_config= self.config['speech_config']
        try:
            self.text_config=self.config['decoder_config']
        except:
            self.text_config = self.config['decoder4_config']
        self.model_config=self.config['model_config']
        self.text_feature=TextFeaturizer(self.text_config)
        self.speech_feature=SpeechFeaturizer(self.speech_config)

        self.init_steps=None
    def update_model_type(self):
        if 'CTC' in self.config['model_config']['name']:
            self.config['decoder_config'].update({'model_type': 'CTC'})
            self.model_type='CTC'
        elif 'Multi' in self.config['model_config']['name']:
            self.config['decoder1_config'].update({'model_type': 'CTC'})
            self.config['decoder2_config'].update({'model_type': 'CTC'})
            self.config['decoder3_config'].update({'model_type': 'CTC'})
            self.config['decoder4_config'].update({'model_type': 'LAS'})
            self.config['decoder_config'].update({'model_type': 'LAS'})
            self.model_type = 'MultiTask'
        elif 'LAS' in self.config['model_config']['name']:
            self.config['decoder_config'].update({'model_type': 'LAS'})
            self.model_type = 'LAS'
        else:
            self.config['decoder_config'].update({'model_type': 'Transducer'})
            self.model_type = 'Transducer'
    def conformer_model(self,training):
        from AMmodel.conformer import ConformerTransducer, ConformerCTC, ConformerLAS
        self.model_config.update({'vocabulary_size': self.text_feature.num_classes})
        if self.model_config['name']=='ConformerTransducer':
            self.model_config.pop('LAS_decoder')
            self.model_config.pop('enable_tflite_convertible')
            self.model_config.update({'speech_config':self.speech_config})
            self.model=ConformerTransducer(**self.model_config)
        elif self.model_config['name']=='ConformerCTC':
            self.model_config.update({'speech_config': self.speech_config})
            self.model=ConformerCTC(**self.model_config)
        elif self.model_config['name']=='ConformerLAS':
            self.config['model_config']['LAS_decoder'].update({'n_classes': self.text_feature.num_classes})
            self.config['model_config']['LAS_decoder'].update({'startid': self.text_feature.start})
            self.model=ConformerLAS(self.config['model_config'], training=training,enable_tflite_convertible=self.config['model_config']['enable_tflite_convertible'],
                                    speech_config=self.speech_config)
        else:
            raise ('not in supported model list')
    def ds2_model(self,training):
        from AMmodel.deepspeech2 import DeepSpeech2CTC,DeepSpeech2LAS,DeepSpeech2Transducer
        self.model_config['Transducer_decoder']['vocabulary_size']= self.text_feature.num_classes
        f,c=self.speech_feature.compute_feature_dim()
        input_shape=[None,f,c]
        self.model_config.update({'input_shape':input_shape})
        if self.model_config['name'] == 'DeepSpeech2Transducer':
            self.model_config.pop('LAS_decoder')
            self.model_config.pop('enable_tflite_convertible')
            self.model = DeepSpeech2Transducer(input_shape,self.model_config,speech_config=self.speech_config)
        elif self.model_config['name'] == 'DeepSpeech2CTC':
            self.model = DeepSpeech2CTC(input_shape,self.model_config,self.text_feature.num_classes,speech_config=self.speech_config)
        elif self.model_config['name'] == 'DeepSpeech2LAS':
            self.model_config['LAS_decoder'].update({'n_classes': self.text_feature.num_classes})
            self.model_config['LAS_decoder'].update({'startid': self.text_feature.start})
            self.model = DeepSpeech2LAS(self.model_config,input_shape, training=training,
                                      enable_tflite_convertible=self.model_config[
                                          'enable_tflite_convertible'],speech_config=self.speech_config)
        else:
            raise ('not in supported model list')
    def multi_task_model(self,training):
        from AMmodel.MultiConformer import ConformerMultiTaskLAS
        token1_feature = TextFeaturizer(self.config['decoder1_config'])
        token2_feature = TextFeaturizer(self.config['decoder2_config'])
        token3_feature = TextFeaturizer(self.config['decoder3_config'])
        token4_feature = TextFeaturizer(self.config['decoder4_config'])

        self.model_config.update({
            'classes1':token1_feature.num_classes,
            'classes2':token2_feature.num_classes,
            'classes3':token3_feature.num_classes,
        })
        self.model_config['LAS_decoder'].update({'n_classes': token4_feature.num_classes})
        self.model_config['LAS_decoder'].update({'startid': token4_feature.start})
        self.model = ConformerMultiTaskLAS(self.model_config,  training=training,
                                    enable_tflite_convertible=self.model_config[
                                        'enable_tflite_convertible'],speech_config=self.speech_config)
    def espnet_model(self,training):
        from AMmodel.espnet import ESPNetCTC,ESPNetLAS,ESPNetTransducer
        self.config['Transducer_decoder'].update({'vocabulary_size': self.text_feature.num_classes})
        if self.model_config['name'] == 'ESPNetTransducer':
            self.model = ESPNetTransducer(self.config,speech_config=self.speech_config)
        elif self.model_config['name'] == 'ESPNetCTC':
            self.model = ESPNetCTC(self.model_config,self.text_feature.num_classes,speech_config=self.speech_config)
        elif self.model_config['name'] == 'ESPNetLAS':
            self.config['LAS_decoder'].update({'n_classes': self.text_feature.num_classes})
            self.config['LAS_decoder'].update({'startid': self.text_feature.start})
            self.model = ESPNetLAS(self.config, training=training,
                                      enable_tflite_convertible=self.config[
                                          'enable_tflite_convertible'],speech_config=self.speech_config)
        else:
            raise ('not in supported model list')
    def load_model(self,training=True):

        if 'ESPNet' in self.model_config['name']:
            self.espnet_model(training)
        elif 'Multi' in self.model_config['name']:
            self.multi_task_model(training)


        elif 'Conformer' in self.model_config['name']:
            self.conformer_model(training)
        else:
            self.ds2_model(training)
        self.model.add_featurizers(self.text_feature)
        f,c=self.speech_feature.compute_feature_dim()


        try:
            if not training:
                if self.text_config['model_type'] != 'LAS':
                    if self.model.mel_layer is not None:
                        self.model._build([3,16000,1])
                        self.model.return_pb_function([None,None,1])
                    else:
                        self.model._build([3, 80, f, c])
                        self.model.return_pb_function([None,None, f, c])

                else:
                    if self.model.mel_layer is not None:
                        self.model._build([3,16000,1], training)
                        self.model.return_pb_function([None,None,1])
                    else:

                        self.model._build([2, 80, f, c], training)
                        self.model.return_pb_function([None,None, f, c])
                self.load_checkpoint(self.config)

        except:
            print('am loading model failed.')
    def convert_to_pb(self,export_path):
        import tensorflow as tf
        concrete_func = self.model.recognize_pb.get_concrete_function()
        tf.saved_model.save(self.model,export_path,signatures=concrete_func)

    def decode_result(self,word):
        de=[]
        for i in word:
            if i!=self.text_feature.stop:
                de.append(self.text_feature.index_to_token[int(i)])
            else:
                break
        return de
    def predict(self,fp):
        if '.pcm' in fp:
            data=np.fromfile(fp,'int16')
            data=np.array(data,'float32')
            data/=32768
        else:
            data = self.speech_feature.load_wav(fp)
        if self.model.mel_layer is None:
            mel=self.speech_feature.extract(data)
            mel=np.expand_dims(mel,0)

            input_length=np.array([[mel.shape[1]//self.model.time_reduction_factor]],'int32')
        else:
            mel=data.reshape([1,-1,1])
            input_length = np.array([[mel.shape[1] // self.model.time_reduction_factor//160]], 'int32')
        result=self.model.recognize_pb(mel,input_length)[0]

        return result

    def load_checkpoint(self,config):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(config['learning_config']['running_config']["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.init_steps= int(files[-1].split('_')[-1].replace('.h5', ''))

if __name__ == '__main__':
    from utils.user_config import UserConfig
    import tensorflow as tf
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    am_config = UserConfig(r'D:\TF2-ASR\configs\am_data.yml', r'D:\TF2-ASR\configs\conformer.yml')
    am=AM(am_config)
    print('load model')
    am.load_model(False)
    print('convert here')
    # am.model.return_pb_function(80, 4)
    # concere = am.model.recognize_pb.get_concrete_function()
    # converter = tf.lite.TFLiteConverter.from_concrete_functions(
    #     [concere]
    # )
    # converter.experimental_new_converter = True
    # # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
    #                                        tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.convert()
    # am.convert_to_pb('./test_model')
    am.convert_to_pb('./test')