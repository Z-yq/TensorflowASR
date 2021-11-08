#include "AM.h"



bool AM::Initialize(const char*  ModelPath)
{
	
	auto model = new cppflow::model(ModelPath);
	
	AMmodel=model;
	cppflow::model& mdl=*AMmodel;
	auto input_wav=cppflow::fill({1,16000,1},1.0f);
	auto input_length=cppflow::fill({1,1},25);
	auto out=mdl({{"serving_default_features:0", input_wav},{"serving_default_length:0",input_length}}, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"});

	std::cout<<"AM init success!!!!!!!!!!!"<<std::endl;
	return AMmodel != nullptr;
	

}

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


}

AM::AM()
{
	AMmodel= nullptr;
}


AM::~AM()
{
}
