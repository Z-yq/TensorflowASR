from AMmodel.model import AM
from LMmodel.trm_lm import LM

class E2EModel(object):
    def __init__(self,am_config,lm_config,):
        self.am=AM(am_config)
        self.lm=LM(lm_config)
    def compile_model(self,training=False):
        self.am.load_model(training)
        self.lm.load_model(training)
        
