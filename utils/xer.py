
# -*- coding: utf-8 -*-

import argparse
import logging
import sys

import numpy as np
logging.basicConfig(
    format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

def levenshtein(u, v):
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    # Operations: (SUB, DEL, INS)
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)], curr_ops[len(v)]

def show_word(r,f):
    if len(r)==len(f)==1:
        r_l=[]
        f_l=[]
        add = 0
        sub = 0
        delect = 0
        if r[0]==f[0]:

            r_l.append('')
            f_l.append('')
        else:
            sub=1
            r_l.append(r[0])
            f_l.append(f[0])
        return add, sub, delect, r_l, f_l
    if len(r)<len(f):

        f_n = np.zeros([2, len(f)])
        num = 0
        num_ = 0
        insert=0
        for fact,i in enumerate(r):
            flag=1
            for j in range(num,len(f)):

                notin=1
                if f[j]==i:
                    if j !=num:
                        test=f[num:j-1]

                        for t in test:
                            if t in r:
                                notin=0
                        if notin:
                            f_n[0][fact] = 1
                            f_n[1][j] = 1
                            flag=0
                            num = j+1
                            break
                    else:
                        f_n[0][fact] = 1
                        f_n[1][j] = 1
                        flag = 0
                        num = j + 1
                        break
            if flag:

                insert+=1
        r_l=[]
        f_l=[]
        xulie=np.where(f_n[1]==1)[0]
        num=0
        for i in range(len(xulie)):
            if i==0:
                f_l.append(f[num:xulie[i]])
            else:
                f_l.append(f[xulie[i-1]+1:xulie[i]])


            if i==len(xulie)-1:
                f_l.append(f[xulie[i]+1:])
        xulie = np.where(f_n[0] == 1)[0]
        num = 0
        for i in range(len(xulie)):
            if i == 0:
                r_l.append(r[num:xulie[i]])
            else:
                r_l.append(r[xulie[i - 1] + 1:xulie[i]])

            if i == len(xulie) - 1:
                r_l.append(r[xulie[i]+1:])
        add=0
        delect=0
        sub=0
        for i,j in zip(r_l,f_l):
            if i==[] and j!=[]:
                add+=len(j)
            elif i!=[] and j==[]:
                delect +=len(i)
            elif i!=[] and j!=[]:
                a=len(i)
                b=len(j)
                if a-b>0:
                    sub+=b
                    delect+=(a-b)
                else:
                    sub+=a
                    add+=(b-a)
    else:
        f_n = np.zeros([2, len(r)])
        num = 0
        num_ = 0
        insert = 0
        for fact, i in enumerate(r):
            flag = 1
            for j in range(num, len(f)):

                notin = 1
                if f[j] == i:
                    if j-num:
                        test = f[num :j-1 ]

                        for t in test:
                            if t in r:
                                notin = 0
                        if notin:
                            f_n[0][fact] = 1
                            f_n[1][j] = 1
                            flag = 0
                            num = j+1
                            break
                    else:
                        f_n[0][fact] = 1
                        f_n[1][j] = 1
                        flag = 0
                        num = j + 1
                        break
            if flag:
                insert += 1
        r_l = []
        f_l = []
        xulie = np.where(f_n[1] == 1)[0]
        num = 0
        for i in range(len(xulie)):
            if i == 0:
                f_l.append(f[num:xulie[i]])
            else:
                f_l.append(f[xulie[i - 1] + 1:xulie[i]])

            if i == len(xulie) - 1:
                f_l.append(f[xulie[i] + 1:])
        xulie = np.where(f_n[0] == 1)[0]
        num = 0
        for i in range(len(xulie)):
            if i == 0:
                r_l.append(r[num:xulie[i]])
            else:
                r_l.append(r[xulie[i - 1] + 1:xulie[i]])

            if i == len(xulie) - 1:
                r_l.append(r[xulie[i] + 1:])
        add = 0
        delect = 0
        sub = 0
        for i, j in zip(r_l, f_l):
            if i == [] and j != []:
                add += len(j)
            elif i != [] and j == []:
                delect += len(i)
            elif i != [] and j != []:
                a = len(i)
                b = len(j)
                if a - b > 0:
                    sub += b
                    delect += (a - b)
                else:
                    sub += a
                    add += (b - a)
    return add,sub,delect,r_l,f_l

def myway(r,h):


    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    for n in range(len(r)):
        # update WER statistics
        i, s, d, r_l, f_l = show_word(r[n].split(), h[n].split())

        wer_s += s
        wer_i += i
        wer_d += d
        wer_n += len(r[n].split())
    return (wer_s + wer_i + wer_d) / wer_n
def wer(r,h):
    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0

    _, (s, d, i) = levenshtein(r, h)

    wer_s += s
    wer_i += i
    wer_d += d
    wer_n += len(r)
    return (wer_s + wer_i + wer_d) / wer_n,s,d,i
if __name__ == '__main__':
    #usage
    # hyp = ["哦我算了",'水电费是']
    # ref = ["哦为是的",'水电费']

    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    num=0
    for n in range(1):
            num+=1

            _, (s, d, i) = levenshtein([1,2,3], [1,2,3,4])

            wer_s += s
            wer_i += i
            wer_d += d
            wer_n += len([1,2,3])


    print('替换:{0},插入:{1},删除:{2},wer:{3},总字数:{4},句子数:{5}'.format(wer_s ,wer_i ,wer_d,(wer_s + wer_i + wer_d) / wer_n,wer_n,num))


