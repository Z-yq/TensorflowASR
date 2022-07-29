import os
with open(r'D:\data\data_aishell\transcript\aishell_transcript_v0.8.txt',encoding='utf-8') as f:
    data=f.readlines()
label_dict={}
for line in data:
    line=line.strip()
    content=line.split(' ')
    key=content[0]
    txt=''.join(content[1:])
    label_dict[key]=txt
for i in os.walk(r'D:\data\data_aishell\wav\train'):
    for j in i[-1]:
        if '.wav' in j:
            key=j.replace('.wav','')
            with open('train.list','a+',encoding='utf-8') as f:
                if key in label_dict:
                    f.write('{}\t{}\n'.format(os.path.join(i[0],j),label_dict[key]))