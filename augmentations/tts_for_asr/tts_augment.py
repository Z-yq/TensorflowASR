import sys
sys.path.append("./")
from tts import TTS
from vc_aug import VC_Aug
import argparse
import random
import soundfile as sf
import os

parser = argparse.ArgumentParser(description="TTS for ASR")
parser.add_argument("-f", "--file", help="text file",required=True)
parser.add_argument("-o", "--out_path", help="output path",required=True)
parser.add_argument(
    "-vn",
    "--voice_num",
    help="voice for TTS. "
    "本次合成每句话需要多少音色",
    type=int,
    default=10,
)
parser.add_argument(
    "-vc",
    "--vc_num",
    help="本次合成每句转换多少个音色",
    type=int,
    default=3,
)

args = parser.parse_args()
os.makedirs(os.path.join(args.out_path, 'wavs'), exist_ok=True)
tts_aug=TTS()
vc_aug=VC_Aug()
with open(args.file,encoding='utf-8') as f:
    data=f.readlines()
index=0

for line in data:
    spks=random.sample(list(range(515)),args.voice_num)
    for spk in spks:

        text=line.strip()
        speed_ratio=1.0
        wav=tts_aug.synthesize(text,spk,speed_ratio)

        sf.write(os.path.join(args.out_path,'wavs',f'{index}.wav'),wav,16000)
        with open(os.path.join(args.out_path,'utterance.txt'),'a+') as f:
            f.write('{}\t{}\n'.format(os.path.join(args.out_path,'wavs',f'{index}.wav'),text))

        if args.vc_num>0:
            vc_spks=random.sample(list(range(1882)),args.vc_num)
            for vc_spk in vc_spks:
                converted_wav=vc_aug.convert(wav,vc_spk)
                sf.write(os.path.join(args.out_path, 'wavs', f'{index}_{vc_spk}.wav'), converted_wav, 16000)
                with open(os.path.join(args.out_path, 'utterance.txt'), 'a+') as f:
                    f.write('{}\t{}\n'.format(os.path.join(args.out_path, 'wavs', f'{index}_{vc_spk}.wav'), text))
        index+=1