
#include "ext/cppflow/cppflow.h"


class LM
{
private:
	cppflow::model * LMmodel = nullptr;

public:
	
	bool Initialize(const char* ModelPath);
	
	std::vector<int64_t> DoInference( const std::vector<int32_t>& InputSequence);
	
	LM();
	~LM();
};

