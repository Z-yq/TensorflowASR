#include <fstream>
#include <iostream>
#include "AM.h"
#include "LM.h"
#include "ext/AudioFile.hpp"
#include "ext/Tokens.hpp"
#include  <time.h>

int main()
{
	/* Demo For RNNT Structure */

	//Init
	AM am ;
	LM lm;
	Tokener AM_Token;
	Tokener LM_Token;
	const char* ampath =  "rnnt_am_saved_model" ;
	const char* lmpath =  "lm_saved_model" ;
	
	AM_Token.load_token("./tokens/am_tokens.txt");
	LM_Token.load_token("./tokens/lm_tokens.txt");

	am.Initialize(ampath);
	lm.Initialize(lmpath);

	std::cout << "Hello TensorflowASR!\n";

	//Read Wav File
	std::cout << "Read File Now...\n";
	std::string wav_path = "./test.wav";
	AudioFile<float> audioFile;
	audioFile.load(wav_path);
	int channel = 0;
	int numSamples = audioFile.getNumSamplesPerChannel();
	std::cout << "Bit Depth: " << audioFile.getBitDepth() << std::endl;
	std::cout << "Sample Rate: " << audioFile.getSampleRate() << std::endl;
	std::cout << "Num Channels: " << audioFile.getNumChannels() << std::endl;
	std::cout << "Length in Seconds: " << audioFile.getLengthInSeconds() << std::endl;
	int all_length = audioFile.getLengthInSeconds() * audioFile.getSampleRate();

	// Prepare AM inputs
	int32_t wav_length = all_length / 640;//640 from the am_data.yml config,CTC Must Check it
    std::vector<float>wav_in;
	for (int i = 0; i < all_length; i++)
	{
		wav_in.push_back(audioFile.samples[channel][i]);
	}
	std::vector<int32_t>length_in;
	length_in.push_back(wav_length);
	
	// Do AM  session run
	
	auto am_out =am.DoInference(wav_in, length_in);

	//get am result to string
	//here 'blank_at_zero=False' in am_data.yml
	std::vector<std::string>am_result;
	
	for (int i = 0; i < am_out.size(); i++)
	{
		int32_t key = am_out[i];
		am_result.push_back(AM_Token.id_to_token[key]);
	
	}
	//Prepare LM inputs
	//here need to check wether the token's id is same with AM,
	//push 'S' id at begin and '/S' id at end
	//if "blank_at_zero = True" in lm_data.yml,so index should +1

	
	std::vector<int32_t>lm_in;
	//CTC structure need to do :
	//********************
	//lm_in.push_back(LM_Token.token_to_id["S"]+1);
	//********************
	for (int i = 0; i < am_out.size(); i++)
	{
		int32_t value = am_out[i]+1;
		lm_in.push_back(value);

	}
	//CTC structure need to do :
	//********************
	//lm_in.push_back(LM_Token.token_to_id["/S"]+1);
	//********************
	// Do LM  session run
	auto lm_out = lm.DoInference(lm_in);
	
	//get lm result to string
	// if "blank_at_zero = True" in lm_data.yml, so index should -1
	std::vector<std::string>lm_result;

	for (int i = 0; i < lm_out.size(); i++)
	{
		int32_t key = lm_out[i]-1;
		lm_result.push_back(LM_Token.id_to_token[key]);

	}
	
	//show result
	
	std::cout << "the AM result:\n";
	for (int i = 0; i < am_result.size(); i++)
	{
		std::cout << am_result[i] << ' ';
	}
	
	std::cout << "\n";
	std::cout << "the LM result:\n";
	for (int i = 0; i <lm_result.size(); i++)
	{
		std::cout << lm_result[i] << ' ';
	}
	
	return 0;

}
