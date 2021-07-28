import tensorflow as tf
import librosa
import numpy as np
import soundfile as sf
def test_20ms_pred():
    model=tf.saved_model.load('vad_model')
    wav = librosa.load('./test.wav', 8000)[0]
    T = (len(wav) // 160) * 160
    wav = wav[:T]
    wav = wav.reshape([-1, 2, 80])
    sil_pred,wav_pred = model.inference(wav)
    wav_pred = wav_pred.numpy().flatten()
    wav_pred/=np.abs(wav_pred).max()
    output = tf.nn.sigmoid(sil_pred)
    vad_pred = output.numpy().flatten()
    T = len(vad_pred) // 10 * 10
    vad_pred = vad_pred[:T]
    vad_pred = vad_pred.reshape([-1, 10])
    vad_pred = np.mean(vad_pred, -1)
    vad_pred = vad_pred.tolist()
    for idx, i in enumerate(vad_pred):
        if i >= 0.5:
            print(idx * 0.1, 's', 'voice')
        else:
            print(idx*0.1,'s','sil')
    sf.write('20ms_model_out.wav',wav_pred,8000)

def test_100ms_pred():
    model=tf.saved_model.load('vad_model')
    wav = librosa.load('./test.wav', 8000)[0]
    T = (len(wav) // 800) * 800
    wav = wav[:T]
    wav = wav.reshape([-1, 10, 80])
    sil_pred,wav_pred = model.inference(wav)
    wav_pred = wav_pred.numpy().flatten()
    wav_pred/=np.abs(wav_pred).max()
    output = tf.nn.sigmoid(sil_pred)
    vad_pred = output.numpy().flatten()
    T = len(vad_pred) // 10 * 10
    vad_pred = vad_pred[:T]
    vad_pred = vad_pred.reshape([-1, 10])
    vad_pred = np.mean(vad_pred, -1)
    vad_pred = vad_pred.tolist()
    for idx, i in enumerate(vad_pred):
        if i >= 0.5:
            print(idx * 0.1, 's', 'voice')
        else:
            print(idx*0.1,'s','sil')
    sf.write('100ms_model_out.wav',wav_pred,8000)
def test_200ms_pred():
    model=tf.saved_model.load('vad_model')
    wav = librosa.load('./test.wav', 8000)[0]
    T = (len(wav) // 1600) * 1600
    wav = wav[:T]
    wav = wav.reshape([-1, 20, 80])
    sil_pred,wav_pred = model.inference(wav)
    wav_pred = wav_pred.numpy().flatten()
    wav_pred/=np.abs(wav_pred).max()
    output = tf.nn.sigmoid(sil_pred)
    vad_pred = output.numpy().flatten()
    T = len(vad_pred) // 10 * 10
    vad_pred = vad_pred[:T]
    vad_pred = vad_pred.reshape([-1, 10])
    vad_pred = np.mean(vad_pred, -1)
    vad_pred = vad_pred.tolist()
    for idx, i in enumerate(vad_pred):
        if i >= 0.5:
            print(idx * 0.1, 's', 'voice')
        else:
            print(idx*0.1,'s','sil')
    sf.write('200ms_model_out.wav',wav_pred,8000)
def test_all_pred():
    model=tf.saved_model.load('vad_model')
    wav = librosa.load('./test.wav', 8000)[0]
    T = (len(wav) // 80) * 80
    wav = wav[:T]
    wav = wav.reshape([1, -1, 80])
    sil_pred,wav_pred = model.inference(wav)
    wav_pred = wav_pred.numpy().flatten()
    wav_pred/=np.abs(wav_pred).max()
    output = tf.nn.sigmoid(sil_pred)
    vad_pred = output.numpy().flatten()
    T = len(vad_pred) // 10 * 10
    vad_pred = vad_pred[:T]
    vad_pred = vad_pred.reshape([-1, 10])
    vad_pred = np.mean(vad_pred, -1)
    vad_pred = vad_pred.tolist()
    for idx, i in enumerate(vad_pred):
        if i >= 0.5:
            print(idx * 0.1, 's', 'voice')
        else:
            print(idx*0.1,'s','sil')
    sf.write('all_model_out.wav',wav_pred,8000)
if __name__ == '__main__':
    test_100ms_pred()

