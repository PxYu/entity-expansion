#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import argparse

cur_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()

dataset = args.dataset


# In[ ]:


fin_path = os.path.join(cur_dir, "data", dataset, "intermediate", "w2v.txt")
fout_path = os.path.join(cur_dir, "data", dataset, "intermediate", "reduced_w2v.txt")

entity2id_path = os.path.join(cur_dir, "data", dataset, "intermediate", "final_entities2eid.json")
entity2id = {}
with open(entity2id_path, 'r') as fin:
    entity2id = json.load(fin)
entity_list = list(entity2id.keys())
eid_list = list(entity2id.values())
len(eid_list)


# In[ ]:


with open(fin_path, 'r') as fin, open(fout_path, 'w+') as fout:
    data = fin.readlines()[1:]
    for line in data:
        if line.split()[0] in entity_list:
            fout.write(line)

