import pypinyin
import Pinyin2Hanzi
import jieba
import numpy as np

dagpms=Pinyin2Hanzi.DefaultDagParams()
txt="而对楼市成交抑制作用最大的限购"
with open('train.list',encoding='utf-8') as f:
    data=f.readlines()
key_words=[]
np.random.shuffle(data)
## mode1
for line in data[:32]:
    txt=line.strip().split('\t')[-1]
    cuts=jieba.lcut(txt)
    for cut in cuts:
        if len(cut)>=2:
            new_word=Pinyin2Hanzi.dag(dagpms,pypinyin.lazy_pinyin(cut))
            for word in new_word:
                word=''.join(word.path)
                if pypinyin.pinyin(word)==pypinyin.pinyin(cut) and cut!=word:
                    key_words.append(word)
                    break

for line in data[:32]:
    txt=line.strip().split('\t')[-1]

    new_word=Pinyin2Hanzi.dag(dagpms,pypinyin.lazy_pinyin(txt))
    for word in new_word:
        word_=''.join(word.path)
        if pypinyin.pinyin(word_)==pypinyin.pinyin(txt) and txt!=word_:
            print(txt)
            print(word.path)
            print('_'*50)
            break
print(key_words)