#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import argparse
import math
from multiprocessing import Process, Manager, Pool, cpu_count, current_process

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("threads")
args = parser.parse_args()
    
dataset = args.dataset
threads = int(args.threads)


# In[ ]:


cur_dir = os.path.join(os.getcwd())
entity2id_path = os.path.join(cur_dir, "data", dataset, "intermediate", "cleaned_entities2eid.json")
eidskipgramcounts_path = os.path.join(cur_dir, "data", dataset, "intermediate", "eidSkipgramCounts.txt")
eid2skipgram_path = os.path.join(cur_dir, "data", dataset, "intermediate", "reduced_eidSkipgramCounts.txt")
eidskipgram_tfidf_path = os.path.join(cur_dir, "data", dataset, "intermediate", "eidSkipgram2TFIDFStrength.txt")
# to create:
write1_path = os.path.join(cur_dir, "data", dataset, "intermediate", "setexpan_eidSkipgramCounts.txt")
write2_path = os.path.join(cur_dir, "data", dataset, "intermediate", "setexpan_eidSkipgram2TFIDFStrength.txt")


# In[ ]:


entity2id = {}
with open(entity2id_path, 'r') as fin:
    entity2id = json.load(fin)
good_entities = list(entity2id.keys())
good_eids = [entity2id[x] for x in good_entities]


# In[ ]:


txt_entity2id_path = os.path.join(cur_dir, "data", dataset, "intermediate", "setexpan_entity2id.txt")
with open(txt_entity2id_path, 'w') as fout:
    for ent in good_entities:
        fout.write("{}\t{}\n".format(ent, entity2id[ent]))


# In[ ]:


with open(eid2skipgram_path, 'r') as fin, open(write1_path, 'w+') as fout:
    data = fin.readlines()
    for line in data:
        splits = line.strip('\n').split('\t')
        if splits[0] in good_eids:
            fout.write(line)


# In[ ]:


freq_index = []
def log_result(result):
    for r in result:
        freq_index.append(r)

def chunk(lst, n):
    p = 0
    cnt = len(lst)
    tmp = []
    for i in range(n-1):
        interval = math.ceil(cnt/(n-i))
        tmp.append(lst[p:p+interval])
        p+=interval
        cnt-=interval
    tmp.append(lst[p:len(lst)])
    return tmp

def runexp(lines, start):
    c = start
    tmp = []
    for line in lines:
        splits = line.strip('\n').split('\t')
        if dataset == "ap89": 
            if int(splits[2])>=2 and splits[0] in good_eids:
                tmp.append(c)
        else:
            if int(splits[2])>=5 and splits[0] in good_eids:
                tmp.append(c)
        c += 1
    return tmp

fin = open(eidskipgramcounts_path, 'r')
lines = fin.readlines()
pool = Pool(threads)
cnt = 0
for lst in chunk(lines, threads):
    print(len(lst), cnt)
    pool.apply_async(runexp, args=(lst, cnt, ), callback=log_result)
    cnt += len(lst)
pool.close()
pool.join()


# In[ ]:


len(freq_index)


# In[ ]:


with open(eidskipgram_tfidf_path, 'r') as fin, open(write2_path, 'w+') as fout:
    data = fin.readlines()
    filtered_lines = [data[x] for x in freq_index]
    for line in filtered_lines:
        fout.write(line)

