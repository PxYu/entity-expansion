#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
    
dataset = args.dataset


# In[ ]:


cur_dir = os.path.join(os.getcwd())
esc_path = os.path.join(cur_dir, "data", dataset, "intermediate", "cleaned_eidskipgramcounts.json")
entity2id_path = os.path.join(cur_dir, "data", dataset, "intermediate", "cleaned_entities2eid.json")
skipgram2id_path = os.path.join(cur_dir, "data", dataset, "intermediate", "skipgram2sid.json")
final_entity2id_path = os.path.join(cur_dir, "data", dataset, "intermediate", "final_entities2eid.json")

esc = {}
entity2id = {}
skipgram2id = {}

with open(esc_path, 'r') as fin:
    esc = json.load(fin)
with open(entity2id_path, 'r') as fin:
    entity2id = json.load(fin)
with open(skipgram2id_path, 'r') as fin:
    skipgram2id = json.load(fin)


# In[ ]:


print("{} entities and {} skipgrams compose {} records.".format(len(entity2id), len(skipgram2id), len(esc)))


# In[ ]:


skipgram2entity = {}
entity2skipgram = {}
for k, v in esc.items():
    splits = k.split(',')
    eid = splits[0]
    sid = splits[1]
    if sid not in skipgram2entity:
        skipgram2entity[sid] = {}
    skipgram2entity[sid][eid] = int(v)
    if eid not in entity2skipgram:
        entity2skipgram[eid] = {}
    entity2skipgram[eid][sid] = int(v)


# In[ ]:


len(skipgram2entity.keys()) == len(skipgram2id)


# In[ ]:


len(entity2skipgram.keys()), len(entity2id)


# In[ ]:


new_entity2id = {}
tmp = list(entity2skipgram.keys())
for k, v in entity2id.items():
    if v in tmp:
        new_entity2id[k] = v
len(new_entity2id)


# In[ ]:


len(skipgram2entity), len(entity2skipgram)


# In[ ]:


with open(final_entity2id_path, 'w+') as fout:
    json.dump(new_entity2id, fout)


# In[ ]:


with open(os.path.join(cur_dir, "data", dataset, "intermediate", "skipgram2entity.json"), 'w+') as fout:
    json.dump(skipgram2entity, fout)
with open(os.path.join(cur_dir, "data", dataset, "intermediate", "entity2skipgram.json"), 'w+') as fout:
    json.dump(entity2skipgram, fout)

