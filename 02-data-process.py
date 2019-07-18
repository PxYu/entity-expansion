#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import copy
import string
import json
import time
import random
import argparse
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
special_chars = ['#', '*', '~', '!', 
                 '@', '$', '%', '^', 
                 '&', '(', ')', '+', 
                 '[', ']', '{', '}', 
                 '.', '/', '-', '<', 
                 '>', '?', ':', ';',
                 '`', ' ']

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
    
dataset = args.dataset


# In[ ]:


def chunk(lst, n):
    p = 0
    cnt = len(lst)
    tmp = []
    for i in range(n-1):
        interval = int(cnt/(n-i)) + 1
        tmp.append(lst[p:p+interval])
        p+=interval
        cnt-=interval
    tmp.append(lst[p:len(lst)])
    return tmp


# In[ ]:


def checkValidString(st):
    for char in special_chars:
        if char in st:
            return False
    if not st[0].isalpha():
        return False
    return True


# In[ ]:


cur_dir = os.path.join(os.getcwd())
entity2id_path = os.path.join(cur_dir, "data", dataset, "intermediate", "entity2id.txt")
eid2skipgram_path = os.path.join(cur_dir, "data", dataset, "intermediate", "reduced_eidSkipgramCounts.txt")
w2v_path = os.path.join(cur_dir, "data", dataset, "intermediate", "w2v.txt")
cleaned_entity2id_path = os.path.join(cur_dir, "data", dataset, "intermediate", "cleaned_entities2eid.json")


# In[ ]:


entity2eid = {}
with open(entity2id_path, 'r', encoding='utf-8') as fin:
    data = fin.readlines()
    for line in data:
        splits = line.split('\t')
        entity2eid[splits[0]] = splits[1].strip('\n')
        
lst = []
for ent in list(entity2eid.keys()):
    if not checkValidString(ent):
        lst.append(ent)

filtered_entity2eid = copy.deepcopy(entity2eid)
for ent in lst:
    del filtered_entity2eid[ent]
    
print(len(entity2eid), len(filtered_entity2eid))


# In[ ]:


w2v_ent = []
with open(w2v_path, 'r') as fin:
    lines = fin.readlines()[1:]
    for line in lines:
        w2v_ent.append(line.split()[0])


# In[ ]:


good_entities = set(list(filtered_entity2eid.keys())).intersection(w2v_ent)


# In[ ]:


good_eids = [entity2eid[x] for x in good_entities]
print(len(good_eids))


# In[ ]:


cleaned_entity2id = {k:filtered_entity2eid[k] for k in good_entities}
len(cleaned_entity2id)


# In[ ]:


with open(cleaned_entity2id_path, 'w+') as fout:
    json.dump(cleaned_entity2id, fout)


# In[ ]:


fin = open(eid2skipgram_path, 'r', encoding='utf-8')
data = fin.readlines()


# In[ ]:


dct = {}
print("#records: " + str(len(lines)))
for line in data:
    seg = line.strip("\r\n").split("\t")
    eid = seg[0]
    skipgram = seg[1]
    count = int(seg[2])
    if skipgram not in dct:
        dct[skipgram] = {}
        dct[skipgram][eid] = count
    else:
        if eid in dct[skipgram]:
            dct[skipgram][eid] += count
        else:
            dct[skipgram][eid] = count
print("#skipgrams: " + str(len(dct)))


# In[ ]:


skipgram2both = {}
for skip in dct:
    mention = 0
    entities = 0
    subdict = dct[skip]
    for ent in subdict:
        entities += 1
        mention += subdict[ent]
    skipgram2both[skip] = [entities, mention]
print(len(skipgram2both))
    
filtered_skipgram2both = {}
for skip in skipgram2both:
    if skipgram2both[skip][0] > 10 and skipgram2both[skip][1] > 30:
        filtered_skipgram2both[skip] = skipgram2both[skip]
print(len(filtered_skipgram2both))
good_skips = [x.strip() for x in list(filtered_skipgram2both.keys())]


# In[ ]:


good_skips[:5]


# In[ ]:


eidskipgramcounts = {}
skipgram2sid = {}
idx = 0
cnt = 0
ttl = 0
start = time.time()
for line in data:
    ttl += 1
    splits = line.strip('\n').split('\t')
    if splits[0] in good_eids and splits[1].strip() in good_skips:
        cnt += 1
        sg = splits[1].strip()
        if sg not in skipgram2sid:
            skipgram2sid[sg] = idx
            eidskipgramcounts["{},{}".format(splits[0], idx)] = int(splits[2])
            idx += 1
        else:
            eidskipgramcounts["{},{}".format(splits[0], skipgram2sid[sg])] = int(splits[2])
        if cnt % 1000 == 0:
            end = time.time()
            print("{} / {} records filtered... {:.2f} seconds elapsed...".format(cnt, ttl, end-start))
            start = time.time()


# In[ ]:


with open('data/{}/intermediate/cleaned_eidskipgramcounts.json'.format(dataset), 'w+') as fout:
    json.dump(eidskipgramcounts, fout)
with open('data/{}/intermediate/skipgram2sid.json'.format(dataset), 'w+') as fout:
    json.dump(skipgram2sid, fout)

