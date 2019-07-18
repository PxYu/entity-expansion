#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import copy
import string
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
    
dataset = args.dataset


# In[ ]:


cur_dir = os.path.join(os.getcwd())
eid2skipgram_path = os.path.join(cur_dir, "data", dataset, "intermediate", "eidSkipgramCounts.txt")
reduced_eis2skipgram_path = os.path.join(cur_dir, "data", dataset, "intermediate", "reduced_eidSkipgramCounts.txt")
fin = open(eid2skipgram_path, 'r', encoding='utf-8')
data = fin.readlines()


# In[ ]:


ttl = 0
cnt = 0
with open(reduced_eis2skipgram_path, 'w+') as fout:
    start = time.time()
    for line in data:
        ttl += 1
        splits = line.strip('\n').split('\t')
        if int(splits[2])>=5:
            cnt += 1
            fout.write(line)
            if cnt % 100000 == 0:
                print("{}/{}...".format(cnt, ttl))
    print("Total time: {:.2f} seconds".format(time.time()-start))

