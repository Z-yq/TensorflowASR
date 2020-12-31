from AMmodel.model import AM,TextFeaturizer,SpeechFeaturizer
from LMmodel.trm_lm import LM

class E2EAM(AM):
    def __init__(self,config):
        self.config = config
        self.model_type='CTC'
        self.speech_config = self.config['speech_config']

        self.text_config = self.config['pinyin_decoder_config']
        self.text_config['model_type']='CTC'
        self.model_config = self.config['am_model_config']
        self.text_feature = TextFeaturizer(self.text_config)
        self.speech_feature = SpeechFeaturizer(self.speech_config)

        self.init_steps = None


    def load_model(self,training=True):
        self.conformer_model(training)

        self.model.add_featurizers(self.text_feature)
        f, c = self.speech_feature.compute_feature_dim()
        try:
            if not training:

                if self.model.mel_layer is not None:
                    self.model._build([3, 16000, 1])
                    self.model.return_pb_function([None, None, 1])
                else:
                    self.model._build([3, 80, f, c])
                    self.model.return_pb_function([None, None, f, c])
                self.load_checkpoint(self.config)
        except:
            print('am loading model failed.')
    def conformer_model(self, training):
        from AMmodel.conformer import  ConformerCTC
        self.model_config.update({'vocabulary_size': self.text_feature.num_classes})
        self.model_config.update({'speech_config': self.speech_config})
        self.model = ConformerCTC(**self.model_config)

class E2ELM(LM):
    def __init__(self,config):
        self.config = config
        self.vocab_featurizer = TextFeaturizer(config['pinyin_decoder_config'])
        self.word_featurizer = TextFeaturizer(config['zh_decoder_config'])
        self.model_config = self.config['lm_model_config']
        self.model_config.update({'input_vocab_size': self.vocab_featurizer.num_classes,
                                  'target_vocab_size': self.word_featurizer.num_classes})


    def load_model(self,training=True):
        from LMmodel.trm_e2e import Transformer
        self.model = Transformer(**self.model_config)
        try:
            if not training:
                self.model._build()
                self.load_checkpoint()
        except:
            print('lm loading model failed.')
        self.model.start_id=self.word_featurizer.start
        self.model.end_id=self.word_featurizer.stop
class E2EModel(object):
    def __init__(self,am_config,lm_config,):
        self.am=E2EAM(am_config)
        self.lm=E2ELM(lm_config)
    def compile_model(self,training=False):
        self.am.load_model(training)
        self.lm.load_model(training)
        
