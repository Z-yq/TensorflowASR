#include <fstream>
#include <iostream>
#include "asr_session.h"
//#include "LM.h"
#include "AudioFile.hpp"
#include  <time.h>




int main()
{

	
	std::cout << "Hello TensorflowASR!\n";

	//Read Wav File
	std::cout << "Read File Now...\n";
	std::string wav_path = "/mnt/j/CODE/ASR/asr_cpp_demo/asr_cpp/test.wav";
	AudioFile<float> audioFile;

	audioFile.load(wav_path);

	int channel = 0;
	int numSamples = audioFile.getNumSamplesPerChannel();
	std::cout << "Bit Depth: " << audioFile.getBitDepth() << std::endl;
	std::cout << "Sample Rate: " << audioFile.getSampleRate() << std::endl;
	std::cout << "Num Channels: " << audioFile.getNumChannels() << std::endl;
	std::cout << "Length in Seconds: " << audioFile.getLengthInSeconds() << std::endl;
	int all_length = audioFile.getLengthInSeconds() * audioFile.getSampleRate();
	// Init Session
	const char* py_txt="./tokens/pinyin.txt";
	const char* hz_txt="./tokens/hanzi.txt";
	const char* punc_ch_txt="./tokens/punc_ch.txt";
	const char* punc_bd_txt="./tokens/punc_bd.txt";
	std::cout << "Init ASR Session.....\n";
	ASR::Session asr_session;
	int dmodel=144;//模型的维度
	int reduction_=640;//samplerate*hop_size*model_reduction
	asr_session.Initial(py_txt,hz_txt, punc_ch_txt,punc_bd_txt,audioFile.getSampleRate(),dmodel);
	asr_session.reduction=reduction_;
	int times = all_length / 1600;
	std::vector<float>wav_in;
	int session_state = 0;

	std::cout << "Start ASR!\n";

	for (int i = 0; i < times; i++)
	{
		for (int j = 0; j < 1600; j++) {
		
			wav_in.push_back(audioFile.samples[channel][i*1600+j]);
		}
		session_state=asr_session.Parase(wav_in);
		wav_in.clear();
		
		if (session_state == 2) {
			std::cout << "asr result:" << asr_session.Get_Asr_Result() << std::endl;
			asr_session.Reset_LIVE_ASR_STATES();
		}
	}
	
	return 0;
	
}
