<h1 align="center">
<p>TensorflowASR</p>
<p align="center">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.2.0-orange">
</p>
</h1>
<h2 align="center">
<p>集成了Tensorflow 2版本的端到端语音识别模型</p>
</h2>
<p align="center">
目前集成了中文的CTC\Transducer\LAS 三种结构
</p>
<p align="center">
当前还在开发阶段
</p>
<p align="center">
欢迎使用并反馈bug
</p>


###[English](https://github.com/Z-yq/TensorflowASR/)|中文版

## Mel Layer

参照librosa库，用TF2实现了语音频谱特征提取的层，这样在跨平台部署时会更加容易。

使用:
- am_data.yml 
   ```
   use_mel_layer: True
   mel_layer_type: Melspectrogram #Spectrogram
   trainable_kernel: True #support train model,not recommend
   ```

## Cpp Inference
C++的demo已经提供。

测试于TensorflowC 2.3.0版本

详细见目录 [cppinference](https://github.com/Z-yq/TensorflowASR/tree/master/CppInference)
## Pretrained Model

所有结果测试于 _`AISHELL TEST`_ 数据集.

**AM:**

Model Name|Mel layer(USE/TRAIN)| link                                          |code|train data        |txt CER|phoneme CER|Model Size|
----------|--------------------|-----------------------------------------------|----|------------------|-------|-----------|---------|
MultiTask |False/False|pan.baidu.com/s/1nDDqcJXBbpFJASYz_U8FfA        |ucqf|aishell2(10 epochs)|10.4   |8.3        |109M|
ConformerRNNT(S)|True/True|pan.baidu.com/s/1bdqeLDBHQ_XmgNuUr6mflw|fqvf|aishell2(10 epochs)|-|9.7|61M|
ConformerCTC(S)|True/True|pan.baidu.com/s/1sh2bUm1HciE6Fu7PHUfRGA|jntv|aishell2(10 epochs)|-|9.9|46M|
ConformerCTC2(S)|True/False|pan.baidu.com/s/12hsjq-lWudeaQzQomV-PDw|ifm6|aishell2(10 epochs)|-|8.1|46M|
ConformerCTC3(S)|False/False|pan.baidu.com/s/1zKDgMHfpOhw10pOSWmtLrQ|gmr5|aishell2(10 epochs)|-|7.0|46M|

**LM:**

Model Name|O2O(Decoder)| link |code|train data|txt cer|model size|params size|
---------|----|------|----|-------|------|----------|-----------|
TransformerO2OE|True(False)|pan.baidu.com/s/1lyqHGacYd7arBrJtlTFdTw|kw0y|aishell2 text(98k steps)|4.4|200M|52M|
TransformerO2OED|True(True)|pan.baidu.com/s/1acvCRpS2j16dxLoCyToB6A|jrfi|aishell2 text(10k steps)|6.2|217M|61M|
Transformer|True(True)|pan.baidu.com/s/1W3HLNNGL3ceJfoxb0P7RMw|qeet|aishell2 text(10k steps)|8.6|233M|61M|
TransformerPunc|False(True)|pan.baidu.com/s/1umwMP2nIzr25NnvG3LTRvw|7ctd|翻译文本|-|76M|30M|

**Speed:**

AM 速度测试(基于Python), 一条约4.1秒的音频 **CPU**响应速度为:

|CTC    |Transducer|LAS  |
|-------|----------|-----|
|150ms  |350ms     |280ms|

LM 速度测试(基于Python),12个字的响应速度 **CPU**:

|O2O-Encoder-Decoder|O2O-Encoder|Encoder-Decoder|
|-------------------|-----------|---------------|
|              100ms|       20ms|          300ms|

**快速使用：**

下载预训练模型，修改 am_data.yml/lm_data.yml 里的目录参数（running_config下的outdir参数），并在修改后的目录中添加 checkpoints 目录，

将model.h5文件放入对应的checkpoints目录中，

修改run-test.py中的读取的config文件（am_data.yml,model.yml）路径，运行run-test.py即可。


## Community
欢迎加入，讨论和分享问题。

<img width="300" height="300" src="./community.jpg">


## What's New?

最新更新

- Change RNNT predict to support C++
- Add C++ Inference Demo,detail in [cppinference](https://github.com/Z-yq/TensorflowASR/tree/master/CppInference)
    

## Supported Structure
-  **CTC**
-  **Transducer**
-  **LAS**
-  **MultiTaskCTC**

## Supported Models

-   **Conformer** 
-   **ESPNet**:`Efficient Spatial Pyramid of Dilated Convolutions`
-   **DeepSpeech2**
-   **Transformer**` 拼音->汉字` 
       -  O2O-Encoder-Decoder `完整的transformer结构，拼音与汉字一一对应的形式
,e.g.: pin4 yin4-> 拼音`
       -  O2O-Encoder `不含decoder部分的结构`
       -  Encoder-Decoder `经典的transformer结构`


## Requirements

-   Python 3.6+
-   Tensorflow 2.2+: `pip install tensorflow`
-   librosa
-   pypinyin `if you need use the default phoneme`
-   keras-bert
-   addons `For LAS structure,pip install tensorflow-addons`
-   tqdm
-   jieba
-   wrap_rnnt_loss `not essential,provide in ./externals`
-   wrap_ctc_decoders `not essential,provide in ./externals`

## Usage

1. 准备train_list.

    **am_train_list** 格式，其中'\t'为tap:

    ```text
    file_path1 \t text1
    file_path2 \t text2
    ……
    ```

    **lm_train_list** 格式:
    ```text
    text1
    text2
    ……
    ```
2. 下载bert的预训练模型，用于LM的辅助训练，如果你不需要LM可以跳过:
            
        https://pan.baidu.com/s/1_HDAhfGZfNhXS-cYoLQucA extraction code: 4hsa
        
3. 修改配置文件 **_`am_data.yml`_** (in ./configs)来设置一些训练的选项，以及修改**model yaml**（如：./configs/conformer.yml） 里的`name`参数来选择模型结构。
4. 然后执行命令:
  
     ```shell
    python train_am.py --data_config ./configs/am_data.yml --model_config ./configs/conformer.yml
    ```
  
5. 想要测试时，可以参考 **_`run-test.py`_** 里写的demo,当然你可以修改 **_`predict`_** 方法来适应你的需求:
     ```python
    from utils.user_config import UserConfig
    from AMmodel.model import AM
    from LMmodel.trm_lm import LM
    
    am_config=UserConfig(r'./configs/am_data.yml',r'./configs/conformer.yml')
    lm_config = UserConfig(r'./configs/lm_data.yml', r'./configs/transformer.yml')
    
    am=AM(am_config)
    am.load_model(training=False)
    
    lm=LM(lm_config)
    lm.load_model()
    
    am_result=am.predict(wav_path)
    if am.model_type=='Transducer':
        am_result =am.decode(am_result[1:-1])
        lm_result = lm.predict(am_result)
        lm_result = lm.decode(lm_result[0].numpy(), self.lm.word_featurizer)
    else:
        am_result=am.decode(am_result[0])
        lm_result=lm.predict(am_result)
        lm_result = lm.decode(lm_result[0].numpy(), self.lm.word_featurizer)
   
    ```
也可以使用**Tester** 来大批量测试数据验证你的模型性能:

第一步需要修改 _`am_data.yml/lm_data.yml`_ 里的 `eval_list` ，格式与  `train_list` 相同

然后执行:
```shell
python eval_am.py --data_config ./configs/am_data.yml --model_config ./configs/conformer.yml
```
该脚本将展示 **SER/CER/DEL/INS/SUB**  几项指标

## Your Model

如果你想加入你自己的模型，你可以将模型加入 `./AMmodel` 目录里 ，声学、语言模型操作都一样，语言模型就放在 `./LMmodel` 里 

```python

from AMmodel.transducer_wrap import Transducer
from AMmodel.ctc_wrap import CtcModel
from AMmodel.las_wrap import LAS,LASConfig
class YourModel(tf.keras.Model):
    def __init__(self,……):
        super(YourModel, self).__init__(……)
        ……
    
    def call(self, inputs, training=False, **kwargs):
       
        ……
        return decoded_feature
        
#To CTC
class YourModelCTC(CtcModel):
    def __init__(self,
                ……
                 **kwargs):
        super(YourModelCTC, self).__init__(
        encoder=YourModel(……),num_classes=vocabulary_size,name=name,
        )
        self.time_reduction_factor = reduction_factor #if you never use the downsample layer,set 1

#To Transducer
class YourModelTransducer(Transducer):
    def __init__(self,
                ……
                 **kwargs):
        super(YourModelTransducer, self).__init__(
            encoder=YourModel(……),
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_lstms=num_lstms,
            lstm_units=lstm_units,
            joint_dim=joint_dim,
            name=name, **kwargs
        )
        self.time_reduction_factor = reduction_factor #if you never use the downsample layer,set 1

#To LAS
class YourModelLAS(LAS):
    def __init__(self,
                ……,
                config,# the config dict in model yml
                training,
                 **kwargs):
        config['LAS_decoder'].update({'encoder_dim':encoder_dim})# encoder_dim is your encoder's last dimension
        decoder_config=LASConfig(**config['LAS_decoder'])

        super(YourModelLAS, self).__init__(
        encoder=YourModel(……),
        config=decoder_config,
        training=training,
        )
        self.time_reduction_factor = reduction_factor #if you never use the downsample layer,set 1

```
然后,将你的模型添加到`./AMmodel/model.py` ,修改方法 `load_model` 来导入你的模型。
## Convert to pb
AM/LM 的操作都相同:
```python
from AMmodel.model import AM
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
