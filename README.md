# TensorflowASR
## 概述
这是一个实验过程中得到的中文端到端语音识别项目

其中包含AM和LM两个部分，由于实验而得，做法相对比较简单，适用性略低。

AM采用了多任务学习训练，分别以：
英文字母为目标
声韵母为目标
拼音为目标

LM采用了Transformer结构，辅以bert特征训练，采取拼音到汉字的做法。
## 进展
1.修改代码成一个系统 OK
2.run-test.py 已经测试通过 OK
3.train_am.py 测试 doing
4.train_lm.py 测试 doing
5.数据预处理模块 doing

## Requirements

Tensorflow 2.2.0
librosa 0.8.0
bert-keras 0.81.0

## 性能指标

测试数据为 AISHELL Test数据集和DEV数据集。
AM以最后的拼音音素为最终结果，以CER（character error rate）为指标。
LM以汉字会结果，同样以CER为指标

AM的性能结果
|Test   |Dev   |
|       |      |
|4.1%   |3.26% |

LM的性能结果
|Test   |Dev   |
|       |      |
|3.12%  |3.16% |

## Usage
可以参考run-test.py里的调用方式。

import hparams
from AMmodel.model import AM
from LMmodel.trm_lm import LM

am=AM(hparams)

lm=LM(hparams)

am_result=am.predict(wav_path)

lm_result=lm.get(am_result)

