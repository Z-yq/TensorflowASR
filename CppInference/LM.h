#include "VoxCommon.hpp"
#include "ext/CppFlow/include/Model.h"


class LM
{
private:
	Model* LMModel;


public:
	
	bool Initialize(const char* ModelPath);
	
	TFTensor<int32_t> DoInference( const std::vector<int32_t>& InputSequence);
	
	LM();
	~LM();
};

