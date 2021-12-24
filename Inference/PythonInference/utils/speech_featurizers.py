
import os
import io
import numpy as np
import librosa
import soundfile as sf



def read_raw_audio(audio, sample_rate=16000):
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        wave = np.asfortranarray(wave)
        if sr != sample_rate:
            wave = librosa.resample(wave, sr, sample_rate)
    elif isinstance(audio, np.ndarray):
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return wave


def normalize_audio_feature(audio_feature: np.ndarray, per_feature=False):
    """ Mean and variance normalization """
    axis = 0 if per_feature else None
    mean = np.mean(audio_feature, axis=axis)
    std_dev = np.std(audio_feature, axis=axis) + 1e-9
    normalized = (audio_feature - mean) / std_dev
    return normalized


def normalize_signal(signal: np.ndarray):
    """ Normailize signal to [-1, 1] range """
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-9)
    return signal * gain


def preemphasis(signal: np.ndarray, coeff=0.97):
    if not coeff or coeff <= 0.0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def deemphasis(signal: np.ndarray, coeff=0.97):
    if not coeff or coeff <= 0.0: return signal
    x = np.zeros(signal.shape[0], dtype=np.float32)
    x[0] = signal[0]
    for n in range(1, signal.shape[0], 1):
        x[n] = coeff * x[n - 1] + signal[n]
    return x


class SpeechFeaturizer:
    def __init__(self, speech_config: dict):

        # Samples
        self.sample_rate = speech_config["sample_rate"]
        self.frame_length = int(self.sample_rate * (speech_config["frame_ms"] / 1000))
        self.frame_step = int(self.sample_rate * (speech_config["stride_ms"] / 1000))
        # Features
        self.num_feature_bins = speech_config["num_feature_bins"]

    def load_wav(self,path):
        wav=read_raw_audio(path,self.sample_rate)
        return wav
    def compute_time_dim(self, seconds: float) -> int:
        # implementation using pad "reflect" with n_fft // 2
        total_frames = seconds * self.sample_rate + 2 * (self.frame_length // 2)
        return int(1 + (total_frames - self.frame_length) // self.frame_step)







