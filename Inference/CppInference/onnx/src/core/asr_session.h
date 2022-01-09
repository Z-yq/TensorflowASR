#include "onnxruntime_cxx_api.h"
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <string>
struct VAD_ONNX{
	const char* model_onnx="./models/vad.onnx";
	Ort::Env vad_env;
	Ort::SessionOptions session_options;
	
	Ort::Session vad_session{vad_env,model_onnx,session_options};

};

struct ENC_ONNX{
	const char* model_onnx="./models/encoder.onnx";
	Ort::Env enc_env;
	Ort::SessionOptions enc_session_options;
	Ort::Session enc_session{enc_env,model_onnx,enc_session_options};

};

struct CTC_ONNX{
	const char* model_onnx="./models/ctc_model.onnx";
	Ort::Env ctc_env;
	Ort::SessionOptions ctc_session_options;
	Ort::Session ctc_session{ctc_env,model_onnx,ctc_session_options};

};

struct TRAN_ONNX{
	const char* model_onnx="./models/translator.onnx";
	Ort::Env trans_env;
	Ort::SessionOptions trans_session_options;
	Ort::Session trans_session{trans_env,model_onnx,trans_session_options};

};

namespace ASR {



	class Tokener {

	public:
		std::string trimstr(std::string s);
		std::map<std::string, int32_t> token_to_id;
		std::map<int32_t, std::string> id_to_token;
		int vocab_size=0;
		void Initial(const char* am_path);


	};


	class Session {
	private:
		VAD_ONNX vad;
		ENC_ONNX encoder;
		CTC_ONNX ctc;
		TRAN_ONNX translator;		

		std::vector<float> AudioBuffer;
		std::vector<float> VADBuffer;

		
		float wavLength = 0.0;
		int chunk_point = 0;
		float vad_point = 0.0;
		int sil_times = 0;
		std::string Session_Name;
		bool vad_result=false;
		int sound_start = 0;
		float voice_start_times;
		float voice_end_times;
		
		std::string asr_results="";
		
	

	public:
		int dmodel=144;
		int reduction=640;
		float samplerate=0.0;
		ASR::Tokener pinyin;
		ASR::Tokener hanzi;
		ASR::Tokener punc_bd;
		ASR::Tokener punc_ch;

		bool Initial(const char* py_txt,const char* hz_txt,const char* punc_ch_txt,const char* punc_bd_txt,int sr,int dmodel_);
		int Parase(std::vector<float> InWav);
		bool VadInference(std::vector<float> InWav);
		std::vector<float> EncoderInference(std::vector<float> InWav);
		std::vector<float> CTCInference(std::vector<float> EncOut);
		std::vector<int> TranslatorInference(std::vector<int> InSeq,std::vector<float> EncOut);
		void Decode_AM_Out(std::vector<int> am_out);
		std::string Get_Asr_Result();
		void Clear_Asr_Result();
		void Reset_Params();
		void Reset_LIVE_ASR_STATES();
		Session();
		~Session();
		std::vector<float> Get_Audio_Buffer();
};

	

	

	

}