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
for filename in os.listdir('../data/eval/cleaned_set/'):
    with open('../data/eval/cleaned_set/'+filename, 'r') as fin:
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


def runsetexpan(setname):
    print("running eval for {}".format(setname))
    for st in good_gold_set:
        if st == setname:
            fin = open('../data/eval/results/{}/{}.log'.format(dataset, st), 'w+')
            entities = good_gold_set[st]
            ap2 = []
            var1 = []
            var2 = []
            times = []
            for numseeds, qs in queries[st].items():
                if numseeds != 3:
                    continue
                start = time.time()
                ap1 = []
                idx = 0
                for q in qs:
                    idx += 1  
                    print("Running \"{}\" [{} seeds] iteration {} ...".format(st, numseeds, idx))
                    samples = [ename2eid[x] for x in list(q)]
                    entities_to_retrieve = [i for i in entities if i not in samples]
                    truth = [eid2ename[int(x)] for x in entities_to_retrieve] # names
                    seedEidsWithConfidence = [(int(ele), 0.0) for ele in samples]
                    fin.write("input: {}\n".format(str([eid2ename[int(x)] for x in samples])))

                    negativeSeedEids = set()
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
                    fin.write("output: {}\n".format(str(answers)))
                    fin.write('----------------------------------------\n')
                    ap1.append(ap(answers, truth))
                end = time.time()
                fin.write("[{} seeds]: MAP: {}, variance: {}, std.variance: {}, avg. time: {:.2f} secs\n\n".format(numseeds, sum(ap1)/len(ap1), np.var(ap1), np.std(ap1), (end-start)/idx))
                fin.write("{}\n".format(str(ap1)))
                ap2.append(sum(ap1)/len(ap1))
                var1.append(np.var(ap1))
                var2.append(np.std(ap1))
                times.append((end-start)/idx)
            fin.write(str(ap2)+'\n')
            fin.write(str(times)+'\n')
            fin.close()

pool = multiprocessing.Pool() 
jobs = [] 
for st in good_gold_set:
    p = multiprocessing.Process(target = runsetexpan, args=(st, ))
    jobs.append(p)
    p.start()
