# TensorflowASR
## 概述
这是一个实验过程中得到的中文端到端语音识别项目

其中包含AM和LM两个部分，由于实验而得，做法相对比较简单，适用性略低。

AM采用了多任务学习训练。

LM采用了Transformer结构，辅以bert特征训练。

## Requirements

Tensorflow 2.2.0
librosa 0.8.0
bert-keras 0.81.0

## 性能指标

测试数据为 AISHELL Test数据集和DEV数据集。以最后的拼音音素为最终结果，以_CER（character error rate）_为指标。

AM
--------------------------
Test         |   Dev     |
--------------------------
4.1%         |   3.26%   |
--------------------------
