

am_train_list='./train_list.txt'
lm_train_list='./train_text.txt'
batch_size=16
save_step=200
# Default Audio hyperparameters


num_mels = 80 #Number of mel-spectrogram channels and local conditioning dimensionality
num_freq = 513 # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
rescale = True #Whether to rescale audio prior to preprocessing
rescaling_max = 0.999 #Rescaling value

use_lws=False #Only used to set as True if using WaveNet, no difference in performance is observed in either cases.
silence_threshold=2 #silence threshold used for sound trimming for wavenet preprocessing

#Mel spectrogram
n_fft = 1024 #Extra window size is filled with 0 paddings to match this parameter
hop_size = int(8000*0.01) #For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
win_size = int(8000*0.05) #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
sample_rate = 8000 #22050 Hz (corresponding to ljspeech dataset) (sox --i <filename>)
frame_shift_ms = None #Can replace hop_size parameter. (Recommended: 12.5)
magnitude_power = 2. #The power of the spectrogram magnitude (1. for energy, 2. for power)

#M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
trim_silence = True #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
trim_fft_size = 2048 #Trimming window size
trim_hop_size = 512 #Trimmin hop length
trim_top_db = 30 #Trimming db difference from reference db (smaller==harder trim.)

#Mel and Linear spectrograms normalization/scaling and clipping
signal_normalization = True #Whether to normalize mel spectrograms to some predefined range (following below parameters)
allow_clipping_in_normalization = True #Only relevant if mel_normalization = True
symmetric_mels = True #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
max_abs_value = 4. #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, not too small for fast convergence)
normalize_for_wavenet = True #whether to rescale to [0, 1] for wavenet. (better audio quality)
clip_for_wavenet = True #whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
wavenet_pad_sides = 1 #Can be 1 or 2. 1 for pad right only, 2 for both sides padding.

#Contribution by @begeekmyfriend
#Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
preemphasize = True #whether to apply filter
preemphasis = 0.97#filter coefficient.

#Limits
min_level_db = -100
ref_level_db = 20
fmin = 55 #Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
fmax = 3800 #To be increased/reduced depending on data.
#redis parameter
use_redis=False
ip='1.127.0.0'
port=6379
data_name='data'

#AM parameters
am_dp=0.
am_layers=3
am_block_units=[768,768,768]
am_save_path='./ckpt/am'
am_dict_file='./AMmodel/am_tokens.txt'

#LM parameter
lm_dict_file='./LMmodel/lm_tokens.txt'
lm_layers=3
lm_d_model=768
lm_heads=8
lm_dff=1024
lm_en_max=1000
lm_de_max=1000
lm_dp=0.

lm_save_path='./ckpt/lm'