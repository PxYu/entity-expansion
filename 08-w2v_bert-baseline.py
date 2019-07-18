#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import random
import copy
import string
import numpy as np
import json
import ast
import argparse
from gensim.models import KeyedVectors
from multiprocessing import Process, Manager, Pool, cpu_count, current_process
cur_dir = os.path.join(os.getcwd())

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("grouped", type=str2bool)
args = parser.parse_args()

dataset = args.dataset
grouped = args.grouped


# In[ ]:


#load w2v
w2v_path = os.path.join(cur_dir, "data", dataset, "intermediate", "w2v.txt")
reduced_w2v_path = os.path.join(cur_dir, "data", dataset, "intermediate", "reduced_w2v.txt")
bert_path = os.path.join(cur_dir, "data", dataset, "intermediate", "bert.txt")

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

# check whether w2v file has proper header
fin = open(w2v_path, 'r+')
lines = fin.readlines()
if len(lines[0].split()) > 2:
    print(lines[0])
    fin.close()
    line_prepender(w2v_path, "{} 100".format(len(lines)))
    print("w2v file modified...")
else:
    print("w2v file is legit...")
    
fin = open(reduced_w2v_path, 'r+')
lines = fin.readlines()
if len(lines[0].split()) > 2:
    print(lines[0])
    fin.close()
    line_prepender(reduced_w2v_path, "{} 100".format(len(lines)))
    print("reduced_w2v file modified...")
else:
    print("reduced_w2v file is legit...")

word_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
bert_vectors = KeyedVectors.load_word2vec_format(bert_path, binary=False)


# In[ ]:


def ap(lst, truth):
    rel = 0
    ap = 0
    for idx in range(1, len(lst)+1):
        if lst[idx-1] in truth:
            rel += 1
            ap += rel/idx
    return ap/len(truth)

def w2v_expansion(seeds):
    w2vsearch = []
    for w in seeds:
        w2vsearch.extend(word_vectors.similar_by_word(w, topn=100))
    w2vsearch = np.array(w2vsearch)
    w2vsearch_sorted = np.sort(w2vsearch,axis=1)[:,1]
    res = []
    counter = 0
    while len(res) <= 100:
        if w2vsearch_sorted[counter] not in seeds and w2vsearch_sorted[counter] not in res:
            res.append(w2vsearch_sorted[counter])
        counter += 1
    return res

def bert_expansion(seeds):
    w2vsearch = []
    for w in seeds:
        w2vsearch.extend(bert_vectors.similar_by_word(w, topn=100))
    w2vsearch = np.array(w2vsearch)
    w2vsearch_sorted = np.sort(w2vsearch,axis=1)[:,1]
    res = []
    counter = 0
    while len(res) <= 100:
        if w2vsearch_sorted[counter] not in seeds and w2vsearch_sorted[counter] not in res:
            res.append(w2vsearch_sorted[counter])
        counter += 1
    return res


# In[ ]:


# data loading
start = time.time()

entity2id_path = os.path.join(cur_dir, "data", dataset, "intermediate", "final_entities2eid.json")
entity2id = {}
with open(entity2id_path, 'r') as fin:
    entity2id = json.load(fin)
entity_list = list(entity2id.keys())

if grouped:
    good_gold_set = {}
    for gname in os.listdir('data/eval/filter_sets/{}/'.format(dataset)):
        for filename in os.listdir('data/eval/filter_sets/{}/{}/'.format(dataset,gname)):
            with open('data/eval/filter_sets/{}/{}/'.format(dataset,gname)+filename, 'r') as fin:
                setname = filename.split('.')[0]
                data = fin.readlines()
                ents = []
                for line in data:
                    ents.append(line.strip('\n'))
                eids = [entity2id[x] for x in ents]
                good_gold_set[setname] = eids

    queries = {}
    for gname in os.listdir('data/eval/queries/{}/'.format(dataset)):
        for filename in os.listdir('data/eval/queries/{}/{}/'.format(dataset,gname)):
            with open('data/eval/queries/{}/{}/'.format(dataset,gname)+filename, 'r') as fin:
                setname = filename.split('.')[0]
                data = fin.readlines()
                queries[setname] = {}
                for idx in range(len(data)):
                    queries[setname][idx+2] = ast.literal_eval(data[idx])
else:
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

    queries = {}
    for filename in os.listdir('data/eval/queries/{}/'.format(dataset)):
        with open('data/eval/queries/{}/'.format(dataset)+filename, 'r') as fin:
            setname = filename.split('.')[0]
            data = fin.readlines()
            queries[setname] = {}
            for idx in range(len(data)):
                queries[setname][idx+2] = ast.literal_eval(data[idx])

def reverse_map(dct):
    tmp = {}
    for k, v in dct.items():
        tmp[v] = k
    return tmp

id2entity = reverse_map(entity2id)

end = time.time()
print("loading data took {:.2f} seconds...".format(end-start))


# In[ ]:


def run_w2v:
    for st in good_gold_set:
        print('=================== {} ===================='.format(st))
        with open("data/eval/results/{}/w2v/{}.log".format(dataset,st), 'w+') as fout:
            entities = good_gold_set[st]
            for numseeds, qs in queries[st].items():
                fout.write("[{} seeds]\n".format(numseeds))
                aps = []
                start = time.time()
                cnt = 0
                for q in qs:
                    cnt += 1
                    samples = q
                    entities_to_retrieve = [i for i in entities if i not in samples]
                    truth = [id2entity[x] for x in entities_to_retrieve]
                    tmp = []
                    for eid in w2v_expansion(samples):
                        if eid not in entity2id:
                            tmp.append("1")
                        else:
                            tmp.append(eid)
                    aps.append(ap(tmp, truth))
                end = time.time()
                print("[{} seeds (w2v)] map: {:.6f} || {:.3f} secs per".format(numseeds, sum(aps)/len(aps), (end-start)/cnt))
                fout.write("{}\n".format(str(aps)))
                fout.write("{}\n".format((end-start)/cnt))

def run_bert:
    for st in good_gold_set:
        print('=================== {} ===================='.format(st))
        with open("data/eval/results/{}/bert/{}.log".format(dataset,st), 'w+') as fout:
            entities = good_gold_set[st]
            for numseeds, qs in queries[st].items():
                fout.write("[{} seeds]\n".format(numseeds))
                aps = []
                start = time.time()
                cnt = 0
                for q in qs:
                    cnt += 1
                    samples = q
                    entities_to_retrieve = [i for i in entities if i not in samples]
                    truth = [id2entity[x] for x in entities_to_retrieve]
                    tmp = []
                    for eid in bert_expansion(samples):
                        if eid not in entity2id:
                            tmp.append("1")
                        else:
                            tmp.append(eid)
                    aps.append(ap(tmp, truth))
                end = time.time()
                print("[{} seeds (BERT)] map: {:.6f} || {:.3f} secs per".format(numseeds, sum(aps)/len(aps), (end-start)/cnt))
                fout.write("{}\n".format(str(aps)))
                fout.write("{}\n".format((end-start)/cnt))


# In[ ]:


pool = Pool()
pool.apply_async(run_w2v, args=(,))
pool.apply_async(run_bert, args=(,))
pool.close()
pool.join()

