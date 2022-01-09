#include "asr_session.h"
#include "ctc_greedy_decoder.h"

std::string ASR::Tokener::trimstr(std::string s)
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
void ASR::Tokener::Initial(const char* am_path)
{
	std::ifstream readFile(am_path);

	std::string s;
	int index = 0;
	while (std::getline(readFile, s))
	{
		s = trimstr(s);
		id_to_token[index] = s;
		token_to_id[s] = index;
		index = index + 1;

	}

	readFile.close();
	std::cout << "token length:\n";
	std::cout << index << std::endl;
	vocab_size=index;
};


bool ASR::Session::VadInference(std::vector<float> InWav) {
	//下面这步为了适配8k的model，如果是16k的vad无需此步。
	//============
	std::vector<float> need_wav;
	for (int i =0;i<InWav.size();i+=2){
		need_wav.push_back(InWav[i]);
	}
	//============

	std::vector<const char*> input_node_names={"inputs"};
	std::vector<const char*> outputs_node_names={"output_0"};
	std::vector<int64_t>  input_node_dims = {1, (int)(need_wav.size()/80),80};
	
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, need_wav.data(),need_wav.size(), input_node_dims.data(),input_node_dims.size());
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));

	auto output_tensors = vad.vad_session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),ort_inputs.size(), outputs_node_names.data(), outputs_node_names.size());

	float* floatarr =output_tensors[0].GetTensorMutableData<float>();
	
	std::vector<float> out;
	for (int i=0;i<(int)(need_wav.size()/80);i++){
		out.push_back(floatarr[i]);
		
	}

	int sum_value=0;
	for (int i = out.size()-10; i < out.size(); i++) {
	
		if (out[i] > -0.1) {
			sum_value++;
		}
	}

	return sum_value > 5;

}	
std::vector<float> ASR::Session::EncoderInference(std::vector<float> InWav) {
	
	std::vector<const char*> input_node_names={"inputs"};
	std::vector<const char*> outputs_node_names={"Identity:0"};
	std::vector<int64_t>  input_node_dims = {1, (int)InWav.size(),1};
	

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, InWav.data(),InWav.size(), input_node_dims.data(),input_node_dims.size());
	
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));
	
	auto output_tensors = encoder.enc_session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),ort_inputs.size(), outputs_node_names.data(), outputs_node_names.size());
	float* floatarr =output_tensors[0].GetTensorMutableData<float>();
	std::vector<float> out;
	for (int i=0;i<(int)(InWav.size()/reduction)*dmodel;i++){
		out.push_back(floatarr[i]);
	}
	return out;
}


std::vector<float> ASR::Session::CTCInference(std::vector<float> EncOut) {
	
	std::vector<const char*> input_node_names={"inputs"};
	std::vector<const char*> outputs_node_names={"Identity:0"};
	std::vector<int64_t>  input_node_dims = {1, (int)(EncOut.size()/dmodel),dmodel};
	

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, EncOut.data(),EncOut.size(), input_node_dims.data(),input_node_dims.size());
	
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));
	
	auto output_tensors = ctc.ctc_session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),ort_inputs.size(), outputs_node_names.data(), outputs_node_names.size());
	float* floatarr =output_tensors[0].GetTensorMutableData<float>();
	std::vector<float> out;
	for (int i=0;i<(int)(EncOut.size()/dmodel)*pinyin.vocab_size;i++){
		out.push_back(floatarr[i]);
		
	}
	
	return out;
}


std::vector<int> ASR::Session::TranslatorInference(std::vector<int> InSeq,std::vector<float> EncOut) {
	
	std::vector<const char*> input_node_names={"inputs","enc"};

	std::vector<const char*> outputs_node_names={"Identity:0"};
	std::vector<int64_t>  input_node_dims1 = {1, (int)InSeq.size()};
	std::vector<int64_t>  input_node_dims2 = {1, (int)(EncOut.size()/dmodel),dmodel};
	auto memory_info1 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	auto memory_info2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor1 = Ort::Value::CreateTensor<int>(memory_info1, InSeq.data(),InSeq.size(),input_node_dims1.data(),input_node_dims1.size());
	Ort::Value input_tensor2 = Ort::Value::CreateTensor<float>(memory_info2, EncOut.data(),EncOut.size(),input_node_dims2.data(),input_node_dims2.size());
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor1));
	ort_inputs.push_back(std::move(input_tensor2));
	auto output_tensors = translator.trans_session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),ort_inputs.size(), outputs_node_names.data(), outputs_node_names.size());
	float* floatarr =output_tensors[0].GetTensorMutableData<float>();
	std::vector<float> out;
	for (int i=0;i<(int)(InSeq.size())*hanzi.vocab_size;i++){
		out.push_back(*floatarr);
		floatarr++;
	}

	std::vector<int> result;
    //argmax和截取有效段
	int steps=InSeq.size();
	for (int i = 0; i < steps; i++) {
		int max_idx = 0;
		float max_prob=-1000.0f;
		for (int j = 0; j <hanzi.vocab_size;j++) {
		if (max_prob < out[j+i*hanzi.vocab_size]) {
			max_idx = j;
			max_prob = out[j+i*hanzi.vocab_size];
		}
		}
		if(max_idx!=1){
		result.push_back(max_idx);}
		else{
			break;
		}
	}

	return result;
}



bool ASR::Session::Initial(const char* py_txt,const char* hz_txt,const char* punc_ch_txt,const char* punc_bd_txt,int sr,int dmodel_){
	pinyin.Initial(py_txt);
	hanzi.Initial(hz_txt);
	punc_bd.Initial(punc_bd_txt);
	punc_ch.Initial(punc_ch_txt);
	samplerate=(float)sr;
	dmodel=dmodel_;
	Reset_Params();
	return true;
}

int ASR::Session::Parase(std::vector<float> InWav) {
	wavLength = wavLength + (float)InWav.size() / samplerate;
	
	for (int i = 0; i < InWav.size(); i++) {
		VADBuffer.push_back((float)InWav[i]);
	}

	if (VADBuffer.size() > 3200) {//VADbuffuer 0.2秒语音
		int buffer_size = VADBuffer.size();
		std::vector<float> VADBuffer_;
		VADBuffer_.assign(VADBuffer.begin() + buffer_size - 3200, VADBuffer.begin() + buffer_size);
		VADBuffer = VADBuffer_;
	}

	if (sound_start) {
		for (int i = 0; i < InWav.size(); i++) {
			AudioBuffer.push_back((float)InWav[i]);
		}
	}
	
	if (wavLength - vad_point >= 0.1) {//每0.1s 做一次VAD
		vad_result = VadInference(VADBuffer);
		vad_point = wavLength;
	}
	if (!sound_start) {
		if (vad_result) {
			
			for (int i = 0; i < VADBuffer.size(); i++) {
				AudioBuffer.push_back(VADBuffer[i]);
			}
			sound_start = 1;
			voice_start_times = wavLength-0.2;
			std::cout << "voice start at " << voice_start_times << " s"<< std::endl;
			return 1;
		}
		else {
			return 0;
		}
		
	}
	else {
		if (!vad_result) {
			sil_times++;
		}
		else {
			sil_times = 0;
		}

		if (sil_times == 5) {//延迟等待0.5秒
			voice_end_times = wavLength - 0.2;
			sound_start = 0;
			std::cout << "voice end at " << voice_end_times<<" s"<< std::endl;
			
			auto enc_out = EncoderInference(AudioBuffer);
			auto ctc_out= CTCInference(enc_out);
			auto out=ctc_greedy_decoder(ctc_out,pinyin.vocab_size-1,pinyin.vocab_size);
			auto result=TranslatorInference(out,enc_out);
			Decode_AM_Out(result);
			AudioBuffer.clear();
			sil_times=0;
			return 2;
		}
		else {
			return 0;
		}

	}


}
void ASR::Session:: Decode_AM_Out(std::vector<int> am_out) {
	
	for (int i = 0; i < am_out.size(); i++)
	{
		asr_results+=hanzi.id_to_token[am_out[i]];
		

	}
	
}

std::string ASR::Session::Get_Asr_Result() {
	return asr_results;
}
void ASR::Session::Clear_Asr_Result() {
	asr_results.clear();
}

void ASR::Session::Reset_LIVE_ASR_STATES() {
	voice_start_times = 0.0;
	voice_end_times = 0.0;
	asr_results = "";

}
void ASR::Session::Reset_Params() {
	wavLength = 0.0;
	chunk_point = 0;
	vad_point = 0.0;
	sil_times = 0;
	Session_Name="";
	vad_result = false;
	sound_start = 0;
	voice_start_times=0.0;
	voice_end_times=0.0;
	asr_results="";
}

std::vector<float> ASR::Session::Get_Audio_Buffer() {
	return AudioBuffer;
}


ASR::Session::Session() {



}
ASR::Session::~Session() {}