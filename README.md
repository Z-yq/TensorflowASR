<h1 align="center">
<p>TiramisuASR</p>
<p align="center">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.2.0-orange">
</p>
</h1>
<h2 align="center">
<p>MultiTask Automatic Speech Recognition in Tensorflow 2</p>
</h2>

<p align="center">
TensorflowASR implements a multitask Chinese speech recognition CTC-based models ,and other models will be enriched in the future.It includes AM and LM. Because of the experiment, the method is relatively simple and the applicability is slightly lower.
Am adopted multi task learning training
English letters as target
Vowels as the target
Pinyin as the target
LM adopts transformer structure, supplemented by Bert feature training and pinyin to Chinese characters.
</p>

## What's New?

-   AM\LM Training script complete

## Supported Models

-   **Multitask-CTCModel** (End2end models using CTC Loss for training)
-   **Transformer** (Pinyin to Chinese characters,LM)


## Requirements

-   Python 3.6+
-   Tensorflow 2.2+: `pip install tensorflow`
-   librosa
-   pypinyin
-   keras-bert

## Usage

Now there is a pre-train model,including ALL open data.

Pre training model: https://pan.baidu.com/s/1_HDAhfGZfNhXS-cYoLQucA extraction code: 4hsa

CKPT in the project directory.

Bert is placed in the LMmodel directory for training.

follow **run-test.py** can use model directly. 


If you want to train own model,

**am_train_list** format:

file_path1 \t text1

file_path2 \t text2

**lm_train_list** format:

text1

text2

modify the path in **hparams.py**,then run **train_am.py** or **train_lm.py**

## Performerce

The test data are aishell test dataset and dev dataset.

Am takes the final Pinyin phoneme as the final result and use CER (character error rate) to test.

LM is based on Chinese characters ,and use cer too.

AM:

|Test   |Dev   |
|-------|------|
|4.1%   |3.26% |

LM:
|Test   |Dev   |
|-------|------|
|3.12%  |3.16% |

AM-LM:

|Test   |Dev   |
|-------|------|
|8.42%  |7.36% |

## Future 
-  Add custom dictionary function
-  Add other end-to-end models
-  Add other language models
