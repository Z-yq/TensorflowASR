
#include <string>
#include <vector>

std::vector<int> ctc_greedy_decoder(const std::vector<float> &probs_seq,int blank_id,int vocab_size) {

  std::vector<int> result;
  std::vector<int> no_repeat;
  int steps=probs_seq.size()/vocab_size;
  for (int i = 0; i < steps; i++) {
    int max_idx = 0;
    float max_prob=-1000.0f;
    for (int j = 0; j <vocab_size;j++) {
      if (max_prob < probs_seq[j+i*vocab_size]) {
        max_idx = j;
        max_prob = probs_seq[j+i*vocab_size];
      }
    }
    result.push_back(max_idx);
  }

  //去重
  for (int i=0;i<result.size();i++)
  {
    if (i == 0)
    {
      no_repeat.push_back(result[i]);
    }
    else if(result[i]!=no_repeat[no_repeat.size()-1])
    {
        no_repeat.push_back(result[i]);
      
    }
  }
  
  //去空白
  result.clear();
  for (int i=0;i<no_repeat.size();i++){
    if (no_repeat[i]!=blank_id){
         result.push_back(no_repeat[i]);
    }
  }
  return result;
}
