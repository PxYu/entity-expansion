#!/home/pxyu/anaconda3/bin/python3

#SBATCH -c 14
#SBATCH --output wiki-mul.log

import os
import sys
sys.path.append(os.getcwd())
import pickle
import util
import set_expan
import time
import numpy as np
import json
import ast
import random
import copy
import multiprocessing
from multiprocessing import Process, current_process

def ap(lst, truth):
    rel = 0
    ap = 0
    for idx in range(1, len(lst)+1):
        if lst[idx-1] in truth:
            rel += 1
            ap += rel/idx
    return ap/len(truth)

FLAGS_USE_TYPE=True
cur_dir = os.path.dirname(os.getcwd())
dataset = "wiki2"
print('dataset:%s' % dataset)
folder = cur_dir+'/data/{}/intermediate/'.format(dataset)
start = time.time()
print('data folder: {}'.format(folder))
print('loading eid and name maps')
eid2ename, ename2eid = util.loadEidToEntityMap(folder+'entity2id.txt')
print('loading eid and skipgram maps')
eid2patterns, pattern2eids = util.loadFeaturesAndEidMap(folder+'reduced_eidSkipgramCounts.txt')
print('loading skipgram strength maps')
eidAndPattern2strength = util.loadWeightByEidAndFeatureMap(folder+'setexpan_eidSkipgram2TFIDFStrength.txt', idx=-1)
print('loading eid and type maps')
eid2types, type2eids = util.loadFeaturesAndEidMap(folder+'eidTypeCounts.txt')
print('loading type strength maps')
eidAndType2strength = util.loadWeightByEidAndFeatureMap(folder+'eidType2TFIDFStrength.txt', idx=-1)
end = time.time()
print("Finish loading all dataset, using %s seconds" % (end-start))

good_gold_set = {}
for filename in os.listdir('../data/eval/filter_sets/'):
    with open('../data/eval/filter_sets/'+filename, 'r') as fin:
        setname = filename.split('.')[0]
        data = fin.readlines()
        ents = []
        for line in data:
            ents.append(line.strip('\n'))
        eids = [ename2eid[x] for x in ents]
        good_gold_set[setname] = eids

queries = {}
for filename in os.listdir('../data/eval/queries/'):
    with open('../data/eval/queries/'+filename, 'r') as fin:
        setname = filename.split('.')[0]
        data = fin.readlines()
        queries[setname] = {}
        for idx in range(len(data)):
            queries[setname][idx+2] = ast.literal_eval(data[idx])

def runlist(lst):
    print(str(current_process().name) + ": " + str(lst))
    for setname in lst:
        runsetexpan(setname)

def runsetexpan(setname):
    for st in good_gold_set:
        if st == setname:
            fin = open('../data/eval/results/{}/setexpan/{}.log'.format(dataset, st), 'w+')
            entities = good_gold_set[st]
            for numseeds, qs in queries[st].items():
                fin.write("[{} seeds]\n".format(numseeds))
                start = time.time()
                ap1 = []
                idx = 0
                for q in qs:
                    idx += 1  
                    print("Running \"{}\" [{} seeds] exp {} ...".format(st, numseeds, idx))
                    samples = [ename2eid[x] for x in list(q)]
                    entities_to_retrieve = [i for i in entities if i not in samples]
                    truth = [eid2ename[int(x)] for x in entities_to_retrieve] # names
                    seedEidsWithConfidence = [(int(ele), 0.0) for ele in samples]
                    negativeSeedEids = set()
                    try:
                        expandedEidsWithConfidence = set_expan.setExpan(
                            seedEidsWithConfidence=seedEidsWithConfidence,
                            negativeSeedEids=negativeSeedEids,
                            eid2patterns=eid2patterns,
                            pattern2eids=pattern2eids,
                            eidAndPattern2strength=eidAndPattern2strength,
                            eid2types=eid2types,
                            type2eids=type2eids,
                            eidAndType2strength=eidAndType2strength,
                            eid2ename=eid2ename,
                            FLAGS_VERBOSE=False,
                            FLAGS_DEBUG=False
                        )
                        answers = []
                        for ele in expandedEidsWithConfidence:
                            answers.append(eid2ename[ele[0]])
                        ap1.append(ap(answers, truth))
                    except:
                        print(setname + " from " + str(current_process().name))
                        ap1.append(0.0)
                end = time.time()
                fin.write("{}\n".format(str(ap1)))
                fin.write("{}\n".format((end-start)/idx))
            fin.close()

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
lst = [x for x in range(len(good_gold_set))]
dis = list(chunks(lst, int(len(lst)/13 + 1)))
names = list(good_gold_set.keys())
            
pool = multiprocessing.Pool() 
jobs = [] 
for ls in dis:
    setnames = [names[i] for i in ls]
    p = multiprocessing.Process(target = runlist, args=(setnames, ))
    jobs.append(p)
    p.start()
