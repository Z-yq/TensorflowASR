import onnxruntime
# import tensorflow as tf
import numpy as np
import time
import os
from punc_recover.models.punc_transformer import positional_encoding
os.environ['CUDA_VISIBLE_DEVICES']='-1'
infer=onnxruntime.InferenceSession('./vad.onnx')
a=positional_encoding(1024,64).numpy()

def creat_mask(seq):
    seq_pad = np.array(seq==0,'float32')
    return seq_pad[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, seq_len)

print([i.name for i in infer.get_inputs()])
print([i.name for i in infer.get_outputs()])
inputs=np.array([[  1 ,117 ,117 , 32 , 46, 201 ,  2]],'int32')
mask=creat_mask(inputs)
length=np.array([7],'float32')
# exit(
data={infer.get_inputs()[0].name:inputs,
      infer.get_inputs()[1].name:mask,
      infer.get_inputs()[2].name:a,}
s=time.time()
out=infer.run([infer.get_outputs()[0].name],input_feed=data)
e=time.time()
print(out[0].argmax(-1),e-s)
