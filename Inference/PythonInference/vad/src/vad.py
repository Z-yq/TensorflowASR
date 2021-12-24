import logging
import os
import onnxruntime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class VAD():
    def __init__(self, config,):
        self.running_config = config['running_config']
        self.model_config = config['model_config']

        self.compile()


    def compile(self):
        self.model=onnxruntime.InferenceSession('./vad/models/vad.onnx')
    def inference(self,wav):
        data = {self.model.get_inputs()[0].name: wav.astype('float32')}

        out = self.model.run([self.model.get_outputs()[0].name], input_feed=data)[0]
        return out









