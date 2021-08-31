#include "LM.h"



bool LM::Initialize(const char*  ModelPath)
{
	auto model = new cppflow::model(ModelPath);
	LMmodel=model;
	cppflow::model& mdl=*LMmodel;
	auto input_seq=cppflow::fill({1,10},1);
	
	auto output=Mdl({{"serving_default_inputs:0",input_seq}}, {"StatefulPartitionedCall:0"});

	std::cout<<"LM init success!!!!!!!!!!!"<<std::endl;
	return LMmodel != nullptr;

}

std::vector<int64_t> LM::DoInference( const std::vector<int32_t>& InputSequence)
{
   

	cppflow::model& mdl=*LMmodel;

	
	std::vector<int64_t> InputSequenceShape = { 1, (int64_t)InputSequence.size()};
	
	auto input_seq=cppflow::tensor(InputSequence,InputSequenceShape);
	auto output=mdl({{"serving_default_inputs:0",input_seq}}, {"StatefulPartitionedCall:0"});
	
	std::cout << "LM Session run success!\n"<<std::endl;
	auto Output = output[0].get_data<int64_t>();
	
	return Output;

}

LM::LM()
{
	LMModel = nullptr;
}


LM::~LM()
{
}
