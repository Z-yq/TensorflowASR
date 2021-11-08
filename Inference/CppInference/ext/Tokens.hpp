#include <fstream>
#include <string>
#include <vector>
#include <map>
std::string trimstr(std::string s)
{
	size_t n = s.find_last_not_of(" \r\n\t");
	if (n != std::string::npos) {
		s.erase(n + 1, s.size() - n);
	}
	n = s.find_first_not_of(" \r\n\t");
	if (n != std::string::npos) {
		s.erase(0, n);
	}
	return s;
}
class Tokener {



	
public:
	std::map<std::string, int32_t> token_to_id;
	std::map<int32_t,std::string> id_to_token;
	void load_token(const char*  am_path);
//TODO:Other function
	
};

void Tokener::load_token(const char* am_path)
{
	std::ifstream readFile(am_path);

	std::string s;
	int index = 0;
	while (std::getline(readFile, s))
	{
		s = trimstr(s);
		id_to_token[index] = s;
		token_to_id[s] = index;
		index=index+1;
		
	}

	readFile.close();
	std::cout << "token length:\n";
	std::cout << index << std::endl;
};
