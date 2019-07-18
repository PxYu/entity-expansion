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

cur_dir = os.path.join(os.getcwd())


# In[ ]:


gold_set = {}
with open('data/gold_set.json', 'r', encoding='utf-8') as fin:
    gold_set = json.load(fin)


# In[ ]:


entity2eid = {}
with open('data/{}/intermediate/final_entities2eid.json'.format(dataset), 'r') as fin:
    entity2eid = json.load(fin)
good_entities = list(entity2eid.keys())


# In[ ]:


new_gold_set = {}
cnt = 0
for setname in gold_set:
    tmp = gold_set[setname]['relevant_entities_all'].keys()
    res = [x.lower().replace(" ", "_") for x in tmp]
    out = [x for x in res if x in good_entities]
    if len(out) >= 7:
        cnt += 1
        new_gold_set[setname] = out
print(len(new_gold_set))


# In[ ]:


new_list = []
with open('data/50sets.txt','r') as fin:
    lines = fin.readlines()
    cnt = 0
    for line in lines:
        if line.strip('\n').strip() in new_gold_set:
            new_list.append(line.strip('\n').strip())
    print("{}/{}".format(len(new_list),len(lines)))


# In[ ]:


for setname in new_list:
    ents = new_gold_set[setname]
    if not os.path.exists('data/eval/filter_sets/{}'.format(dataset)):
        os.makedirs('data/eval/filter_sets/{}'.format(dataset))
    with open('data/eval/filter_sets/{}/{}.set'.format(dataset, setname), 'w+') as fout:
        for ent in ents:
            fout.write(ent+"\n")


# In[ ]:


# set names of eval sets that could complement existing sets
renewable_lists = ['C020', 'C023', 'C027', 'C058', 'C083', 'C089', 'C013', 'C062']
renewable_names = [gold_set[x]['title'] for x in renewable_lists]
corres_names = ['fruits', 'colors', 'elements', 'animals', 'body', 'instruments', 'states', 'transportation']


# In[ ]:


for idx in range(len(renewable_lists)):
    with open('data/eval/set/merge/{}.set'.format(corres_names[idx])) as fin:
        tmp = list(gold_set[renewable_lists[idx]]['relevant_entities_all'].keys())
        x = [i.lower() for i in tmp]
        lines = fin.readlines()
        y = []
        for line in lines:
            y.append(line.strip().lower().replace(" ", "_"))
        outs = [e for e in list(set(x).union(y)) if e in good_entities]
        if len(outs) >= 7:
            with open('data/eval/filter_sets/{}/{}.set'.format(dataset,renewable_lists[idx]), 'w+') as fout:
                for ent in outs:
                    fout.write(ent+"\n")


# In[ ]:


for setname in ['grains', 'family_members', 'vegetables', 'days']:
    with open('data/eval/set/standalone/{}.set'.format(setname)) as fin:
        lines = fin.readlines()
        tmp = []
        for line in lines:
            tmp.append(line.strip().lower().replace(" ", "_"))
        outs = [e for e in tmp if e in good_entities]
        outss = list(set(outs))
        if len(outss) >= 7:
            with open('data/eval/filter_sets/{}/{}.set'.format(dataset,setname), 'w+') as fout:
                for ent in outss:
                    fout.write(ent+"\n")

