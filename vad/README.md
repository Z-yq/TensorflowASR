# VAD

该 VAD 训练于8k数据，具有一定的抗噪能力和语音增强能力。

模型的输入为wav原始信号，最低为10ms的采样点 80，输入shape为 **[1,1,80]**

详情执行该目录下的online_vad.py和offline_vad.py查看效果。