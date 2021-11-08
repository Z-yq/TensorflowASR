<h1 align="center">
<p>TensorflowASR</p>
<p align="center">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.2.0-orange">
</p>
</h1>
<h2 align="center">
<p>集成了Tensorflow 2版本的端到端语音识别模型，并且RTF(实时率)在0.1左右</p>
</h2>
<p align="center">
当前branch为V2版本，为CTC+translate结构
</p>
<p align="center">

[V1版本](https://github.com/Z-yq/TensorflowASR/tree/master)

</p>
<p align="center">
欢迎使用并反馈bug
</p>




## 实现功能

- VAD+降噪
- 在线流式识别/离线识别
- 标点恢复

## 其它项目

TTS：https://github.com/Z-yq/TensorflowTTS

NLU:  -

BOT:  -


## Mel Layer

参照librosa库，用TF2实现了语音频谱特征提取的层。

或者可以使用更小参数量的[Leaf](https://github.com/google-research/leaf-audio) 。

使用:
- am_data.yml 
   ```
   mel_layer_type: Melspectrogram #Spectrogram/leaf
   trainable_kernel: True #support train model,not recommend
   ```

## Cpp Inference
于2021.08.31 更新了C++的demo。

测试于TensorflowC 2.5.0版本

详细见目录 [cppinference](https://github.com/Z-yq/TensorflowASR/tree/master/CppInference)



# Streaming Conformer

现在支持流式的Conformer结构啦，同epoch训练下，和全局conformer的CER仅差0.8%。

![streaming_conformer](asr/streaming_model.svg)

## Pretrained Model

所有结果测试于 _`AISHELL TEST`_ 数据集.

**RTF**(实时率) 测试于**CPU**单核解码任务。 


**AM:**

Model Name|Mel layer(USE/TRAIN)| link                                          |code|train data        |phoneme CER(%)|Params Size|RTF
----------|--------------------|-----------------------------------------------|----|------------------|:---------:|:-------:|-----
ConformerCTC(M)|True/False|pan.baidu.com/s/1NPk17DUr0-lBgwCkC5dFuQ|7qmd|aishell-1(20 epochs)| 6.2/5.1|32M|0.114
ConformerCTS(S)|True/False|pan.baidu.com/s/1mHR2RryT7Rw0D4I9caY0QQ|7g3n|aishell-1(20 epochs)| 9.1/8.7|10M|0.056
StreamingConformerCTC|True/False|pan.baidu.com/s/1NAmkIUqO5dWM2AvL_3xlVw|d1u6|aishell-1(10 epochs)| 10.1 |15M|0.08



**LM:**

Model Name|O2O(Decoder)| link |code|train data|txt cer|model size|params size|RTF|
---------|----|------|----|-------|------|----------|-----------|-----|
TransformerO2OE|True(False)|pan.baidu.com/s/1X11OE_sk7yNTjtDpU7sfvA|sxrw|aishell-1 text(30 epochs)|4.4|43M|10M|0.06|
TransformerO2OED|True(True)|pan.baidu.com/s/1acvCRpS2j16dxLoCyToB6A|jrfi|aishell2 text(10k steps)|6.2|217M|61M|0.13|

**Punc:**
Model Name|O2O(Decoder)| link |code|train data|txt cer|model size|params size|RTF|
---------|----|------|----|-------|------|----------|-----------|-----|
PuncModel|True(False)|pan.baidu.com/s/1b_6eKEWfL50pmvuS7ZRimg|47f5|NLP开源数据|-|38M|10M|0.005|

**快速使用：**

run-test.py中默认的模型为 ：

AM: ConformerCTC(M)

LM：TransformerO2OE

Punc：PuncModel

全部下载，放置于代码目录下即可运行。



指定运行模型：

修改 am_data.yml/lm_data.yml 里的目录参数（running_config下的outdir参数），并在修改后的目录中添加 checkpoints 目录，

将model_xx.h5(xx为数字)文件放入对应的checkpoints目录中，

修改run-test.py中的读取的config文件（am_data.yml,model.yml）路径，运行run-test.py即可。


## Community
欢迎加入，讨论和分享问题。 群1已满。

<img width="300" height="300" src="./community.jpg">


## What's New?

最新更新

- :1st_place_medal: [2021.08.19]更改了Streaming Conformer结构，舍弃了之前的LSTM结构以提升训练速度，目前已经验证推举配置的训练结果只和全局的conformer相差1%左右。

- 增加了标点恢复的模型和预训练模型
- 优化了一些逻辑
- 添加了C++ 接口 Demo，详见 [cppinference](https://github.com/Z-yq/TensorflowASR/tree/master/CppInference)
  

## Supported Structure
-  **CTC**+**Streaming**


## Supported Models

-   **Conformer** 
-   **StreamingConformer**


## Requirements

-   Python 3.6+
-   Tensorflow 2.5+: `pip install tensorflow`
-   librosa
-   pypinyin `if you need use the default phoneme`
-   keras-bert
-   addons `For LAS structure,pip install tensorflow-addons`
-   tqdm


## Usage

1. 准备train_list和test_list.

    **asr_train_list** 格式，其中'\t'为tap:

    ```text
    file_path1 \t text1
    file_path2 \t text2
    ……
    ```

    例如：
    ```text
    /opt/data/test.wav	这个是一个例子
    ```
   
    
以下为vad和标点恢复的训练数据准备格式（非必需）：
    
   **vad_train_list** 格式:
   
   ```text
   wav_path1
   wav_path2
   ……
   ```
   例如：
   ```text   
这是一个例子
```

   
   
   
   **punc_train_list**格式：
   
   ```text
    text1
    text2
    ……
   ```
   同LM的格式，每行的text包含标点，目前标点只支持每个字后跟一个标点，连续的标点视为无效。
   
   比如：
   ```text
   这是：一个例子哦。 √(正确格式）
   
   这是：“一个例子哦”。 ×(错误格式）
   
   这是：一个例子哦“。 ×(错误格式）

```
  
   
2. 下载bert的预训练模型，用于标点恢复模型的辅助训练，如果你不需要标点恢复可以跳过:
            
   
        https://pan.baidu.com/s/1_HDAhfGZfNhXS-cYoLQucA extraction code: 4hsa
    
3. 修改配置文件 **_`am_data.yml`_** (./asr/configs)来设置一些训练的选项，以及修改**model yaml**（如：./asr/configs/conformer.yml） 里的`name`参数来选择模型结构。

4. 然后执行命令:
  
     ```shell
    python train_asr.py --data_config ./asr/configs/am_data.yml --model_config ./asr/configs/ConformerS.yml
    ```
  
5. 想要测试时，可以参考 **_`./asr/run-test.py`_** 里写的demo,当然你可以修改 **_`stt`_** 方法来适应你的需求:
   ```python
    python ./asr/run_test.py  
   ```
也可以使用**Tester** 来大批量测试数据验证你的模型性能:



执行:
```shell
python eval_am.py --data_config ./asr/configs/am_data.yml --model_config ./asr/configs/ConformerS.yml
```
该脚本将展示 **SER/CER/DEL/INS/SUB**  几项指标


6.训练VAD或者标点恢复模型，请参照以上步骤。


## Convert to pb
AM/LM 的操作都相同:
```python
from asr.models import AM
am_config = UserConfig('...','...')
am=AM(am_config)
am.load_model(False)
am.convert_to_pb(export_path)
```
## Tips
如果你想用你自己的音素，需要对应 `am_dataloader.py/lm_dataloader.py` 里的转换方法。

```python
def init_text_to_vocab(self):#keep the name
    
    def text_to_vocab_func(txt):
        return your_convert_function

    self.text_to_vocab = text_to_vocab_func #here self.text_to_vocab is a function,not a call
```

不要忘记你的音素列表用 **_`S`_** 和 **_`/S`_** 打头,e.g:

        S
        /S
        de
        shì
        ……




## References

感谢关注：


https://github.com/usimarit/TiramisuASR `modify from it`

https://github.com/noahchalifour/warp-transducer

https://github.com/PaddlePaddle/DeepSpeech

https://github.com/baidu-research/warp-ctc

## Licence

允许并感谢您使用本项目进行学术研究、商业产品生产等，但禁止将本项目作为商品进行交易。

Overall, Almost models here are licensed under the Apache 2.0 for all countries in the world.

Allow and thank you for using this project for academic research, commercial product production, allowing unrestricted commercial and non-commercial use alike. 

However, it is prohibited to trade this project as a commodity.
