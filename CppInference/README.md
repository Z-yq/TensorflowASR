## CppInference

Demo For Tensorflow C to run pb file.

Test On Tensorflow C 2.3.0

Mel_layer for AM feature extractor.


## Support Structure

- Mel_layer+ConformerCTC

- Mel_layer+ConformerTransducer

-   Transformer
       -  O2O-Encoder-Decoder 
       -  O2O-Encoder 
       -  Encoder-Decoder 

## TensorflowC

First to install Tensorflow C 

Follow guide in https://tensorflow.google.cn/install/lang_c?hl=en

Need gcc >= 7.5



## Usage

step 1: Train and test your Model using python.
```shell
python train_am.py
python train_lm.py
```
step 2: Convert to pb.

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

am.convert_to_pb(am_export_path)
lm.convert_to_pb(lm_export_path)

```

step 3: Use **`saved_model_cli`** to show the model arch.
```shell
saved_model_cli show --dir lm_export_path --all
```

it will show like this:
```text

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_INT32
        shape: (-1, -1)
        name: serving_default_inputs:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_0'] tensor_info:
        dtype: DT_INT32
        shape: (-1, -1)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict
```
Get 

Node |tensor name|tensor dtype|
-----|---------|------|
inputs|serving_default_inputs|DT_INT32|
outputs|StatefulPartitionedCall|DT_INT32|

Here,AM model will include `inputs1/inputs2`


step 4: Change The Code in AM.cpp/LM.cpp 

```text

TFTensor<int32_t> AM::DoInference(const std::vector<float>& InWav, const std::vector<int32_t>& InputLength)
{
    Tensor input_wavs{ Mdl,"serving_default_features" };//inputs1 node name
	std::vector<int64_t> InputWavShape = { 1, (int64_t)InWav.size(),1 };
	input_wavs.set_data(InWav, InputWavShape);
	Tensor input_length{ Mdl,"serving_default_length" };//inputs2 node name
	std::vector<int64_t> InputLengthShape = { 1, 1 };
	input_length.set_data(InputLength, InputLengthShape);
	
	Tensor out_result{ Mdl,"StatefulPartitionedCall" };//output Node name
	TFTensor<int32_t> Output = VoxUtil::CopyTensor<int32_t>(out_result);
	
	return Output;
}
```
*notice: ``TFTensor<T>``, should be same with Node dtype. 

step 5: Follow what's in the **`CppInference.cpp`** , and you'll get it.

Be care: am_tokens.txt and lm_tokens.txt should  be consistent with the files you use for training.

****
_am_data.yml  And lm_data.yml ,The 'blank_at_zero' attribute of is better to be the same, otherwise you need to map according to the rules._ 
****

## Simple Compile
```text
g++ CppInference.cpp AM.cpp LM.cpp ./ext/CppFlow/src/Tensor.cpp ./ext/CppFlow/src/Model.cpp -ltensorflow -o CppInference

./CppInference

```

It will recognize `test.wav`,and print result on screen.

Can use `python_predict.py` to check .

### Reference

Thanks:

- Tensorflow C API: https://www.tensorflow.org/install/lang_c

- CppFlow (TF C API -> C++ wrapper): https://github.com/serizba/cppflow

- AudioFile (for WAV Read): https://github.com/adamstark/AudioFile
