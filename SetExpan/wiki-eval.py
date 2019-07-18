import pickle
import os
import util
import set_expan
import time
import os
import json
import random
import copy
import sys

def ap(lst, truth):
    rel = 0
    ap = 0
    for idx in range(1, len(lst)+1):
        if lst[idx-1] in truth:
            rel += 1
            ap += rel/idx
    return ap/len(truth)

FLAGS_USE_TYPE=True
cur_dir = os.path.dirname(os.path.realpath(__file__))
data = "wiki"
print('dataset:%s' % data)
folder = '/../../data/'+data+'/intermediate/'
start = time.time()
print('loading eid and name maps')
eid2ename, ename2eid = util.loadEidToEntityMap(cur_dir + folder+'entity2id.txt') #entity2id.txt
print('loading eid and skipgram maps')
eid2patterns, pattern2eids = util.loadFeaturesAndEidMap(cur_dir + folder+'reduced_eidSkipgramCounts.txt') #eidSkipgramCount.txt
print('loading skipgram strength map')
eidAndPattern2strength = util.loadWeightByEidAndFeatureMap(cur_dir + folder+'setexpan_eidSkipgram2TFIDFStrength.txt', idx=-1) #(eid, feature, weight) file
print('loading eid and type maps')
eid2types, type2eids = util.loadFeaturesAndEidMap(cur_dir + folder+'eidTypeCounts.txt') #eidTypeCount.txt
print('loading type strength map')
eidAndType2strength = util.loadWeightByEidAndFeatureMap(cur_dir + folder+'eidType2TFIDFStrength.txt', idx=-1) #(eid, feature, weight) file
end = time.time()
print("Finish loading all dataset, using %s seconds" % (end-start))

good_gold_set = {}
for filename in os.listdir('../../data/eval/cleaned_set/'):
    with open('../../data/eval/cleaned_set/'+filename, 'r') as fin:
        setname = filename.split('.')[0]
        data = fin.readlines()
        ents = []
        for line in data:
            ents.append(line.strip('\n'))
        eids = [ename2eid[x] for x in ents]
        good_gold_set[setname] = eids

setname = sys.argv[1]
print("running eval for {}".format(setname))
## Start set expansion
for st in good_gold_set:
    if st == setname:
        fin = open('../../data/eval/results/{}.log'.format(st), 'w')
        entities = good_gold_set[st]
        ap2 = []
        for numseeds in range(2,6):
            start = time.time()
            ap1 = []
            for idx in range(1, 31):
                print("running \"{}\" [{} seeds] iteration {}".format(st, numseeds, idx))
                samples = random.sample(entities, numseeds)
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
                    # print("eid=", ele[0], "ename=", eid2ename[ele[0]])
                    answers.append(eid2ename[ele[0]])
                fin.write("output: {}\n".format(str(answers)))
                fin.write('----------------------------------------\n')
                ap1.append(ap(answers, truth))
            end = time.time()
            fin.write("[{} seeds]: MAP: {}, avg. time: {:.2f} secs\n\n".format(numseeds, sum(ap1)/len(ap1), (end-start)/30))
            ap2.append(sum(ap1)/len(ap1))
        fin.write(str(ap2)+'\n')
        fin.close()