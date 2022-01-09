# CPPInference

该为基于ONNX的C++ ASR方案。

主要逻辑在asr_session.cpp中,完成了VAD\ASR相关功能。

# Usage

需要cmake  

编译

```shell
sh build.sh
```

会在该目录下生成test.out 

使用：
```
export LD_LIBRARY_PATH=./ext/onnxruntime/lib:$LD_LIBRARY_PATH 
./test.out
```

该demo会识别当前路径下的test.wav
