import os
from utils import ljqpy
import h5py
import numpy as np
def replace(txt, mod=1):
    if mod == 1:
        table = {f: t for t, f in zip(
            u'，。！？“”（）％＃＠＆：',
            u',.!?【】()%#@&:')}
        for i in table.keys():
            txt = txt.replace(i, table[i])
    elif mod == 2:
        r1 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        txt = re.sub(r1, '', txt)

    return txt



class ITokens():
    def __init__(self, vocab):
        self.vocab = vocab
        self.t2id, self.id2t = self.make_dict()

    def make_dict(self):
        vocal_dict = dict()
        inver_dict = dict()
        for index, i in enumerate(self.vocab):
            vocal_dict[i] = index
            inver_dict[index] = i
        return vocal_dict, inver_dict

    def id(self, x):
        return self.t2id[x]

    def num(self):
        return len(self.vocab)

def only_chinese(word):
    txt = ''
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            txt += ch

    return txt

class TokenList:
    def __init__(self, token_list, model='stt'):
        if model == 'stt':
            self.id2t = token_list
        elif model=='lm':
            self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>'] + token_list
        else:
            self.id2t = ['P', 'S', '/S']+token_list
        self.t2id = {v: k for k, v in enumerate(self.id2t)}

    def id(self, x):
        return self.t2id.get(x, 1)

    def token(self, x):
        return self.id2t[x]

    def num(self):
        return len(self.id2t)

    def startid(self):
        return self.t2id['S']

    def endid(self):
        return self.t2id['/S']

    def padid(self):
        return self.t2id['P']


def pad_to_longest(xs, tokens, max_len=999):
    longest = min(len(max(xs, key=len)) + 2, max_len)
    X = np.zeros((len(xs), longest), dtype='int32')
    X[:, 0] = tokens.startid()
    for i, x in enumerate(xs):
        x = x[:max_len - 2]
        for j, z in enumerate(x):
            X[i, 1 + j] = tokens.id(z)
        X[i, 1 + len(x)] = tokens.endid()
    return X


def MakeS2SDict(fn=None, min_freq=1, delimiter=' ', dict_file=None, model='stt'):
    if dict_file is not None and os.path.exists(dict_file):
        print('loading', dict_file)
        lst = ljqpy.LoadList(dict_file)
        midpos = lst.index('<@@@>')
        itokens = TokenList(lst[:midpos], model=model)
        otokens = TokenList(lst[midpos + 1:], model=model)
        return itokens, otokens
    data = ljqpy.LoadCSV(fn)
    wdicts = [{}, {}]
    for ss in data:
        for seq, wd in zip(ss, wdicts):
            for w in seq.split(delimiter):
                wd[w] = wd.get(w, 0) + 1
    wlists = []
    for wd in wdicts:
        wd = ljqpy.FreqDict2List(wd)
        wlist = [x for x, y in wd if y >= min_freq]
        wlists.append(wlist)
    print('seq 1 words:', len(wlists[0]))
    print('seq 2 words:', len(wlists[1]))
    itokens = TokenList(wlists[0], model)
    otokens = TokenList(wlists[1], model)
    if dict_file is not None:
        ljqpy.SaveList(wlists[0] + ['<@@@>'] + wlists[1], dict_file)
    return itokens, otokens


def MakeS2SData(fn=None, itokens=None, otokens=None, delimiter=' ', h5_file=None, max_len=200):
    if h5_file is not None and os.path.exists(h5_file):
        print('loading', h5_file)
        with h5py.File(h5_file) as dfile:
            X, Y = dfile['X'][:], dfile['Y'][:]
        return X, Y
    data = ljqpy.LoadCSVg(fn)
    Xs = [[], []]
    for ss in data:
        for seq, xs in zip(ss, Xs):
            xs.append(list(seq.split(delimiter)))
    X, Y = pad_to_longest(Xs[0], itokens, max_len), pad_to_longest(Xs[1], otokens, max_len)
    if h5_file is not None:
        with h5py.File(h5_file, 'w') as dfile:
            dfile.create_dataset('X', data=X)
            dfile.create_dataset('Y', data=Y)
    return X, Y


def S2SDataGenerator(fn, itokens, otokens, batch_size=64, delimiter=' ', max_len=999):
    Xs = [[], []]
    while True:
        for ss in ljqpy.LoadCSVg(fn):
            for seq, xs in zip(ss, Xs):
                xs.append(list(seq.split(delimiter)))
            if len(Xs[0]) >= batch_size:
                X, Y = pad_to_longest(Xs[0], itokens, max_len), pad_to_longest(Xs[1], otokens, max_len)
                yield [X, Y], None
                Xs = [[], []]


if __name__ == '__main__':
    import pypinyin as ppy
    from tqdm import tqdm
    import re
    itokens, otokens = MakeS2SDict('data/my.txt', dict_file='data/stt.txt', model='stt')
    new=[]
    with open('/opt/zhongyuqi/Data/MAGICDATA/text_dict.txt', encoding='utf-8') as f:
        txt_dict = f.read()
        txt_dict = eval(txt_dict)
    for key in tqdm(txt_dict.keys()):
        txt=txt_dict[key]
        txt = txt.replace(' ', '')
        txt = txt.replace('、', '')
        txt = txt.replace(',', '')
        txt = replace(txt, 2)
        txt = only_chinese(txt)
        s=ppy.lazy_pinyin(txt,3)
        m=ppy.lazy_pinyin(txt,9)
        new.extend(s)
        new.extend(m)
        new=list(set(new))
    print(new)

