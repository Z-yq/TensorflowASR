
#include "ext/cppflow/cppflow.h"

class AM
{
private:
	cppflow::model * AMmodel = nullptr;


public:
	
	bool Initialize(const char* ModelPath);
	
	
	std::vector<int64_t> DoInference(const std::vector<float> InWav, const std::vector<int32_t> InputLength);
	
	AM();
	~AM();
};

