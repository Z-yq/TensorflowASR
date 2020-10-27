import tensorflow as tf
import librosa
import numpy as np
import time
model=tf.saved_model.load('./rnnt_am_saved_model','serve')
wav=librosa.load('test.wav',16000)[0]
wav=wav.reshape([1,-1,1])
length=np.array([[wav.shape[1]//640]])
print(model.recognize_pb(wav,length))

