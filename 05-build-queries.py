#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import math
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
    
dataset = args.dataset

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

cur_dir = os.path.join(os.getcwd())


# In[ ]:


good_gold_set = {}
for filename in os.listdir('data/eval/filter_sets/{}/'.format(dataset)):
    with open('data/eval/filter_sets/{}/'.format(dataset)+filename, 'r') as fin:
        setname = filename.split('.')[0]
        data = fin.readlines()
        ents = []
        for line in data:
            ents.append(line.strip('\n'))
        good_gold_set[setname] = ents


# In[ ]:


ttl = 0
print("{} sets".format(len(good_gold_set)))
for st in good_gold_set:
    print("===={}====".format(st))
    if not os.path.exists('data/eval/queries/{}'.format(dataset)):
        os.makedirs('data/eval/queries/{}'.format(dataset))
    with open('data/eval/queries/{}/{}.query'.format(dataset,st), 'w') as fout:
        c = 0
        for i in range(2,6):
            if dataset == "ap89":
                leng = min(nCr(len(good_gold_set[st]),i), 50)
            else:
                leng = min(nCr(len(good_gold_set[st]),i), 100)
            c += leng
            samples = []
            while len(samples) < leng:
                tmp = tuple(sorted(random.sample(good_gold_set[st],i)))
                if tmp not in samples:
                    samples.append(tmp)
            fout.write(str(samples)+"\n")
        ttl += c
        print(c)


# In[ ]:


print(ttl)

