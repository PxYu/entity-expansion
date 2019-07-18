#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import time
import math
import nltk
import copy
import random
import ast
import argparse
from gensim.models import KeyedVectors

SM = 2 # 1 for log, 2 for sqrt
parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
    
dataset = args.dataset


# In[ ]:


'''
utility functions
'''
def smooth(num):
    if SM == 1:
        return math.log2(num)
    elif SM == 2:
        return math.sqrt(num)
    return num

def getTokenized(string):
    return "_".join(nltk.word_tokenize(string)).lower()

def getRankedEntityBySkipgram(skip, topk=None):
    '''
    input: skipgramid, topk (cutoff)
    output: [(entity name, count), ...]
    '''
    ret = []
    dct = skipgram2entity[skip]
    if topk == None:
        for x in sorted(dct, key=dct.get, reverse=True):
            ret.append((id2entity[x], dct[x]))
    else:
        for x in sorted(dct, key=dct.get, reverse=True)[:topk]:
            ret.append((id2entity[x], dct[x]))
    return ret

def getEntityWeightInSkipgram(eid, skipid):
    '''
    input: entity-id, skipgram-id
    output: #ent&skip / SIGMA(#ent_k&skip) 
    '''
    skip = skipgram2entity[skipid]
    if eid not in skip:
        return 0
    else:
        counts = 0
        for k, v in skip.items():
            counts += v
        return skip[eid]/counts
    
def getSkipgramWeightInEntity(eid, skipid):
    '''
    input: entity-id, skipgram-id
    output: #ent&skip / SIGMA(#ent&skip_k) 
    '''
    ent = skipgram2entity[eid]
    if skipid not in ent:
        return 0
    else:
        counts = 0
        for k, v in ent.items():
            counts += v
        return ent[skipid] / counts

def getListOfEntityWeightInSkipgram(eidlst, skipid):
    skip = skipgram2entity[skipid]
    counts = 0
    for k, v in skip.items():
        counts += v
    head = 0
    for eid in eidlst:
        head += skip[eid]
    return head / counts

def ap(lst, truth):
    rel = 0
    ap = 0
    for idx in range(1, len(lst)+1):
        if lst[idx-1] in truth:
            rel += 1
            ap += rel/idx
    return ap/len(truth)

def aggregateDictionaries(lst):
    dct = {}
    for d in lst:
        for ent, rank in d.items():
            if ent not in dct:
                dct[ent] = rank
            else:
                dct[ent] += rank
    return dct

def getSkipgramMentions(skip):
    skip_dct = skipgram2entity[skip]
    cnt = 0
    for k, v in skip_dct.items():
        cnt += v
    return cnt


# In[ ]:


cur_dir = os.path.join(os.getcwd())

# data loading
start = time.time()

skipgram2entity_path = os.path.join(cur_dir, "data", dataset, "intermediate", "skipgram2entity.json")
skipgram2entity = {}
with open(skipgram2entity_path, 'r') as fin:
    skipgram2entity = json.load(fin)
    
skipgram2id_path = os.path.join(cur_dir, "data", dataset, "intermediate", "skipgram2sid.json")
skipgram2id = {}
with open(skipgram2id_path, 'r') as fin:
    skipgram2id = json.load(fin)
    
eid2skipgram_path = os.path.join(cur_dir, "data", dataset, "intermediate", "entity2skipgram.json")
eid2skipgram = {}
with open(eid2skipgram_path, 'r') as fin:
    eid2skipgram = json.load(fin)
    
entity2id_path = os.path.join(cur_dir, "data", dataset, "intermediate", "final_entities2eid.json")
entity2id = {}
with open(entity2id_path, 'r') as fin:
    entity2id = json.load(fin)
entity_list = list(entity2id.keys())

good_gold_set = {}
for filename in os.listdir('data/eval/filter_sets/{}/'.format(dataset)):
    with open('data/eval/filter_sets/{}/'.format(dataset)+filename, 'r') as fin:
        setname = filename.split('.')[0]
        data = fin.readlines()
        ents = []
        for line in data:
            ents.append(line.strip('\n'))
        eids = [entity2id[x] for x in ents]
        good_gold_set[setname] = eids

entity_count = {}
for skip, vals in skipgram2entity.items():
    for eid, mentions in vals.items():
        if eid not in entity_count:
            entity_count[eid] = mentions
        else:
            entity_count[eid] += mentions
            
queries = {}
for filename in os.listdir('data/eval/queries/{}/'.format(dataset)):
    with open('data/eval/queries/{}/'.format(dataset)+filename, 'r') as fin:
        setname = filename.split('.')[0]
        data = fin.readlines()
        queries[setname] = {}
        for idx in range(len(data)):
            queries[setname][idx+2] = ast.literal_eval(data[idx])

w2v_path = os.path.join(cur_dir, "data", dataset, "intermediate", "reduced_w2v.txt")
w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=False)

# bert_path = os.path.join(cur_dir, "data", dataset, "intermediate", "bert.txt")
# bert = KeyedVectors.load_word2vec_format(bert_path, binary=False)
    
def reverse_map(dct):
    tmp = {}
    for k, v in dct.items():
        tmp[v] = k
    return tmp

id2skipgram = reverse_map(skipgram2id)
id2entity = reverse_map(entity2id)

end = time.time()
print("loading data took {:.2f} seconds...".format(end-start))


# In[ ]:


def getIntersectionEntityCounts(lst, norm):
    '''
    input: list of eids
    output: ranked list of entities by counts
    '''
    dct = {}
    skipgram = [eid2skipgram[x] for x in lst]
    inter = list(set(skipgram[0]).intersection(*skipgram[1:]))
    for x in inter:
        entities = getRankedEntityBySkipgram(x, 100)
        for ent in entities:
            if not norm:
                if ent[0] not in dct:
                    dct[ent[0]] = ent[1]
                else:
                    dct[ent[0]] += ent[1]
            else:
                if ent[0] not in dct:
                    dct[ent[0]] = smooth(ent[1])
                else:
                    dct[ent[0]] += smooth(ent[1])
    ret = sorted(dct, key=dct.get, reverse=True)
    for seed in lst:
        if id2entity[seed] in ret:
            ret.remove(id2entity[seed])
    return ret

def getDoubleWeightedIntersectionEntityCounts(lst, norm = True, w = 0):
    '''
    input: list of eids
    output: ranked list of entities by counts
    "double" means defining a d-weight based on skipgram features (#entities and #mentions)
    '''
    dct = {}
    skipgram = [eid2skipgram[x] for x in lst]
    inter = list(set(skipgram[0]).intersection(*skipgram[1:]))
    for x in inter:
        weight = getListOfEntityWeightInSkipgram(lst, x)
        entities = getRankedEntityBySkipgram(x, 100)
        if w == 0:
            weight2 = 1
        if w == 1:
            weight2 = math.sqrt((1/len(skipgram2entity[x])) * (1/getSkipgramMentions(x)))
        elif w == 2:
            weight2 = math.sqrt(1/len(skipgram2entity[x]))
        elif w == 3:
            weight2 = math.sqrt((1/getSkipgramMentions(x)))
        elif w == 4:
            weight2 = 1/len(skipgram2entity[x])
        elif w == 5:
            weight2 = 1/getSkipgramMentions(x)
        for ent in entities:
            if not norm:
                if ent[0] not in dct:
                    dct[ent[0]] = ent[1] * weight * weight2
                else:
                    dct[ent[0]] += ent[1] * weight * weight2
            else:
                if ent[0] not in dct:
                    dct[ent[0]] = smooth(ent[1]) * weight * weight2
                else:
                    dct[ent[0]] += smooth(ent[1]) * weight * weight2
    ret = sorted(dct, key=dct.get, reverse=True)
    for seed in lst:
        if id2entity[seed] in ret:
            ret.remove(id2entity[seed])
    return ret


# In[ ]:


def getUnionEntityCounts(lst, norm):
    dct = {}
    skipgram = [eid2skipgram[x] for x in lst]
    uni = set().union(*skipgram)
    for x in uni:
        for y in lst:
            weight = getEntityWeightInSkipgram(y, x)
            if weight != 0:
                entities = getRankedEntityBySkipgram(x, 100)
                for ent in entities:
                    if norm:
                        if ent[0] not in dct:
                            dct[ent[0]] = smooth(ent[1])
                        else:
                            dct[ent[0]] += smooth(ent[1])
                    else:
                        if ent[0] not in dct:
                            dct[ent[0]] = ent[1]
                        else:
                            dct[ent[0]] += ent[1]
    ret = sorted(dct, key=dct.get, reverse=True)
    for seed in lst:
        if id2entity[seed] in ret:
            ret.remove(id2entity[seed])
    return ret

def getDoubleWeightedUnionEntityCounts(lst, norm=True, w = 0):
    dct = {}
    skipgram = [eid2skipgram[x] for x in lst]
    uni = set().union(*skipgram)
    for x in uni:
        if w == 0:
            weight2 = 1
        if w == 1:
            weight2 = math.sqrt((1/len(skipgram2entity[x])) * (1/getSkipgramMentions(x)))
        elif w == 2:
            weight2 = math.sqrt(1/len(skipgram2entity[x]))
        elif w == 3:
            weight2 = math.sqrt((1/getSkipgramMentions(x)))
        elif w == 4:
            weight2 = 1/len(skipgram2entity[x])
        elif w == 5:
            weight2 = 1/getSkipgramMentions(x)
        for y in lst:
            weight = getEntityWeightInSkipgram(y, x)
            if weight != 0:
                entities = getRankedEntityBySkipgram(x, 100)
                for ent in entities:
                    if not norm:
                        if ent[0] not in dct:
                            dct[ent[0]] = ent[1] * weight * weight2
                        else:
                            dct[ent[0]] += ent[1] * weight * weight2
                    else:
                        if ent[0] not in dct:
                            dct[ent[0]] = smooth(ent[1]) * weight * weight2
                        else:
                            dct[ent[0]] += smooth(ent[1]) * weight * weight2
    ret = sorted(dct, key=dct.get, reverse=True)
    for seed in lst:
        if id2entity[seed] in ret:
            ret.remove(id2entity[seed])
    return ret


# In[ ]:


print("Exp4: union with...\n")
for st in good_gold_set:
    print('=================== {} ===================='.format(st))
    with open("data/eval/results/{}/ecase/{}.log".format(dataset,st), 'w+') as fout:
        entities = good_gold_set[st]
        for numseeds, qs in queries[st].items():
            fout.write("[{} seeds]\n".format(numseeds))
            ap7 = []
            cnt = 0
            start = time.time()
            for q in qs:  
                samples = [entity2id[x] for x in list(q)]
                entities_to_retrieve = [i for i in entities if i not in samples]
                truth = [id2entity[x] for x in entities_to_retrieve]
                ap7.append(ap(getDoubleWeightedUnionEntityCounts(samples, True), truth))
                cnt += 1
            end = time.time()
            print("[{} seeds] | {:.6f} || {:.3f} secs per".format(numseeds, sum(ap7)/len(ap7), (end-start)/cnt))
            fout.write("{}\n".format(str(ap7)))
            fout.write("{}\n".format((end-start)/cnt))


# In[ ]:


def getUnionEntityCountsWithW2V(lst, norm, power = 2):
    dct = {}
    skipgram = [eid2skipgram[x] for x in lst]
    uni = set().union(*skipgram)
    for x in uni:
        for y in lst:
            weight = getEntityWeightInSkipgram(y, x)
            if weight != 0:
                entities = getRankedEntityBySkipgram(x, 100)
                for ent in entities:
                    w2v_weight = w2v.similarity(w1=ent[0], w2=id2entity[y]) ** power
                    if norm:
                        if ent[0] not in dct:
                            dct[ent[0]] = smooth(ent[1]) * w2v_weight
                        else:
                            dct[ent[0]] += smooth(ent[1]) * w2v_weight
                    else:
                        if ent[0] not in dct:
                            dct[ent[0]] = ent[1] * w2v_weight
                        else:
                            dct[ent[0]] += ent[1] * w2v_weight
    ret = sorted(dct, key=dct.get, reverse=True)
    for seed in lst:
        if id2entity[seed] in ret:
            ret.remove(id2entity[seed])
    return ret

def getDoubleWeightedUnionEntityCountsWithW2V(lst, norm=True, w = 0, power = 2):
    dct = {}
    skipgram = [eid2skipgram[x] for x in lst]
    uni = set().union(*skipgram)
    for x in uni:
        if w == 0:
            weight2 = 1
        if w == 1:
            weight2 = math.sqrt((1/len(skipgram2entity[x])) * (1/getSkipgramMentions(x)))
        elif w == 2:
            weight2 = math.sqrt(1/len(skipgram2entity[x]))
        elif w == 3:
            weight2 = math.sqrt((1/getSkipgramMentions(x)))
        elif w == 4:
            weight2 = 1/len(skipgram2entity[x])
        elif w == 5:
            weight2 = 1/getSkipgramMentions(x)
        for y in lst:
            weight = getEntityWeightInSkipgram(y, x)
            if weight != 0:
                entities = getRankedEntityBySkipgram(x, 100)
                for ent in entities:
                    w2v_weight = w2v.similarity(w1=ent[0], w2=id2entity[y]) ** power
                    if not norm:
                        if ent[0] not in dct:
                            dct[ent[0]] = ent[1] * weight * weight2 * w2v_weight
                        else:
                            dct[ent[0]] += ent[1] * weight * weight2 * w2v_weight
                    else:
                        if ent[0] not in dct:
                            dct[ent[0]] = smooth(ent[1]) * weight * weight2 * w2v_weight
                        else:
                            dct[ent[0]] += smooth(ent[1]) * weight * weight2 * w2v_weight
    ret = sorted(dct, key=dct.get, reverse=True)
    for seed in lst:
        if id2entity[seed] in ret:
            ret.remove(id2entity[seed])
    return ret


# In[ ]:


print("Exp5: union with w2v...\n")
for st in good_gold_set:
    print('=================== {} ===================='.format(st))
    with open("data/eval/results/{}/ecase_w2v/{}.log".format(dataset,st), 'w+') as fout:
        entities = good_gold_set[st]
        for numseeds, qs in queries[st].items():
            fout.write("[{} seeds]\n".format(numseeds))
            ap7 = []
            cnt = 0
            start = time.time()
            for q in qs:  
                samples = [entity2id[x] for x in list(q)]
                entities_to_retrieve = [i for i in entities if i not in samples]
                truth = [id2entity[x] for x in entities_to_retrieve]
                ap7.append(ap(getDoubleWeightedUnionEntityCountsWithW2V(samples, True, 0, 7), truth))
                cnt += 1
            end = time.time()
            fout.write("{}\n".format(str(ap7)))
            fout.write("{}\n".format((end-start)/cnt))


# In[ ]:


bert_path = os.path.join(cur_dir, "data", dataset, "intermediate", "bert.txt")
bert = KeyedVectors.load_word2vec_format(bert_path, binary=False)


# In[ ]:


def getUnionEntityCountsWithBert(lst, norm, power = 2):
    dct = {}
    skipgram = [eid2skipgram[x] for x in lst]
    uni = set().union(*skipgram)
    for x in uni:
        for y in lst:
            weight = getEntityWeightInSkipgram(y, x)
            if weight != 0:
                entities = getRankedEntityBySkipgram(x, 100)
                for ent in entities:
                    w2v_weight = bert.similarity(w1=ent[0], w2=id2entity[y]) ** power
                    if norm:
                        if ent[0] not in dct:
                            dct[ent[0]] = smooth(ent[1]) * w2v_weight
                        else:
                            dct[ent[0]] += smooth(ent[1]) * w2v_weight
                    else:
                        if ent[0] not in dct:
                            dct[ent[0]] = ent[1] * w2v_weight
                        else:
                            dct[ent[0]] += ent[1] * w2v_weight
    ret = sorted(dct, key=dct.get, reverse=True)
    for seed in lst:
        if id2entity[seed] in ret:
            ret.remove(id2entity[seed])
    return ret

def getDoubleWeightedUnionEntityCountsWithBert(lst, norm=True, w = 0, power = 2):
    dct = {}
    skipgram = [eid2skipgram[x] for x in lst]
    uni = set().union(*skipgram)
    for x in uni:
        if w == 0:
            weight2 = 1
        if w == 1:
            weight2 = math.sqrt((1/len(skipgram2entity[x])) * (1/getSkipgramMentions(x)))
        elif w == 2:
            weight2 = math.sqrt(1/len(skipgram2entity[x]))
        elif w == 3:
            weight2 = math.sqrt((1/getSkipgramMentions(x)))
        elif w == 4:
            weight2 = 1/len(skipgram2entity[x])
        elif w == 5:
            weight2 = 1/getSkipgramMentions(x)
        for y in lst:
            weight = getEntityWeightInSkipgram(y, x)
            if weight != 0:
                entities = getRankedEntityBySkipgram(x, 100)
                for ent in entities:
                    w2v_weight = bert.similarity(w1=ent[0], w2=id2entity[y]) ** power
                    if not norm:
                        if ent[0] not in dct:
                            dct[ent[0]] = ent[1] * weight * weight2 * w2v_weight
                        else:
                            dct[ent[0]] += ent[1] * weight * weight2 * w2v_weight
                    else:
                        if ent[0] not in dct:
                            dct[ent[0]] = smooth(ent[1]) * weight * weight2 * w2v_weight
                        else:
                            dct[ent[0]] += smooth(ent[1]) * weight * weight2 * w2v_weight
    ret = sorted(dct, key=dct.get, reverse=True)
    for seed in lst:
        if id2entity[seed] in ret:
            ret.remove(id2entity[seed])
    return ret


# In[ ]:


print("Exp7: union with bert...\n")
for st in good_gold_set:
    print('=================== {} ===================='.format(st))
    with open("data/eval/results/{}/ecase_bert/{}.log".format(dataset,st), 'w+') as fout:
        entities = good_gold_set[st]
        for numseeds, qs in queries[st].items():
            fout.write("[{} seeds]\n".format(numseeds))
            ap7 = []
            cnt = 0
            start = time.time()
            for q in qs:  
                samples = [entity2id[x] for x in list(q)]
                entities_to_retrieve = [i for i in entities if i not in samples]
                truth = [id2entity[x] for x in entities_to_retrieve]
                ap7.append(ap(getDoubleWeightedUnionEntityCountsWithBert(samples, True, 0, 7), truth))
                cnt += 1
            end = time.time()
            fout.write("{}\n".format(str(ap7)))
            fout.write("{}\n".format((end-start)/cnt))

