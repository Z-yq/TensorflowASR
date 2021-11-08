## CppInference

Demo For Tensorflow C to run pb file.

Test On **Tensorflow C 2.5.0/Ubuntu 18.04/Centos 7**

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
from asr.models import AM
from punc_recover.models.trm_lm import LM

am_config=UserConfig(r'./configs/data.yml',r'./configs/conformer.yml')
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

std::vector<int64_t> AM::DoInference(const std::vector<float> InWav, const std::vector<int32_t> InputLength)
{

	
	cppflow::model& mdl=*AMmodel;
	
	std::vector<int64_t> InputWavShape = { 1, (int64_t)InWav.size(),1 };
	std::vector<int64_t> InputLengthShape = { 1, 1 };
	auto input_wav=cppflow::tensor(InWav,InputWavShape);
	std::cout<<"input wav:\n"<< input_wav<<std::endl;
	auto input_length=cppflow::tensor(InputLength,InputLengthShape);
	std::cout<<"input length:\n"<< input_length<<std::endl;
	auto output = mdl({{"serving_default_features:0", input_wav},{"serving_default_length:0",input_length}}, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"});
	std::cout << "AM Session run success!\n"<<std::endl;

	auto Output = output[0].get_data<int64_t>();
	
	return Output;

```
*notice: ``std::vector<T>``, should be same with Node dtype. 

step 5: Follow what's in the **`CppInference.cpp`** , and you'll get it.

Be care: am_tokens.txt and lm_tokens.txt should  be consistent with the files you use for training.

****
_am_data.yml  And lm_data.yml ,The 'blank_at_zero' attribute of is better to be the same, otherwise you need to map according to the rules._ 
****

## Simple Compile
```text
g++ CppInference.cpp AM.cpp LM.cpp -I/yourdir/tensorflow_c/include -L/yourdir/tensorflow_c/lib -ltensorflow -o CppInference

./CppInference

```

It will recognize `test.wav`,and print result on screen.

Can use `python_predict.py` to check .

### Reference

Thanks:

- Tensorflow C API: https://www.tensorflow.org/install/lang_c

- CppFlow (TF C API -> C++ wrapper): https://github.com/serizba/cppflow

- AudioFile (for WAV Read): https://github.com/adamstark/AudioFile
