#include "LM.h"



bool LM::Initialize(const char*  ModelPath)
{
	try {
		LMModel = new Model(ModelPath);
	
	}
	catch (...) {
		LMModel = nullptr;
		return false;

	}
	return true;

}

TFTensor<int32_t> LM::DoInference( const std::vector<int32_t>& InputSequence)
{
    VX_IF_EXCEPT(!LMModel, "Tried to infer LMmodel on uninitialized model!!!!");

	Model& Mdl = *LMModel;

	Tensor input_seqs{ Mdl,"serving_default_inputs" };
	std::vector<int64_t> InputSequenceShape = { 1, (int64_t)InputSequence.size()};
	input_seqs.set_data(InputSequence, InputSequenceShape);
	
	
	Tensor out_result{ Mdl,"StatefulPartitionedCall" };
	LMModel->run(input_seqs, out_result);
	std::cout << "LM Session run success!\n"<<std::endl;

	TFTensor<int32_t> Output = VoxUtil::CopyTensor<int32_t>(out_result);
	
	return Output;


}

LM::LM()
{
	LMModel = nullptr;
}


LM::~LM()
{
}
