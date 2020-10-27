#include "AM.h"



bool AM::Initialize(const char*  ModelPath)
{
	try {
		AMModel = new Model(ModelPath);
	
	}
	catch (...) {
		AMModel = nullptr;
		return false;

	}
	return true;

}

TFTensor<int32_t> AM::DoInference(const std::vector<float>& InWav, const std::vector<int32_t>& InputLength)
{
    VX_IF_EXCEPT(!AMModel, "Tried to infer AMmodel on uninitialized model!!!!");

	
	Model& Mdl = *AMModel;

	Tensor input_wavs{ Mdl,"serving_default_features" };
	std::vector<int64_t> InputWavShape = { 1, (int64_t)InWav.size(),1 };
	input_wavs.set_data(InWav, InputWavShape);
	Tensor input_length{ Mdl,"serving_default_length" };
	std::vector<int64_t> InputLengthShape = { 1, 1 };
	input_length.set_data(InputLength, InputLengthShape);
	
	Tensor out_result{ Mdl,"StatefulPartitionedCall" };
	std::vector<Tensor*> inputs = { &input_wavs,&input_length };
	AMModel->run(inputs, out_result);
	std::cout << "AM Session run success!\n"<<std::endl;

	TFTensor<int32_t> Output = VoxUtil::CopyTensor<int32_t>(out_result);
	
	return Output;


}

AM::AM()
{
	AMModel= nullptr;
}


AM::~AM()
{
}
