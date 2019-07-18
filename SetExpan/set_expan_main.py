'''
__author__: Jiaming Shen, Ellen Wu
__description__: The main function for running SetExpan algorithm
__latest_update__: 10/12/2017
'''
import util
import set_expan
import time
import os
import json
import random

curr_dir = os.path.dirname(os.path.realpath(__file__))

## Setting global versions
FLAGS_USE_TYPE=False

## Loading Corpus
data = "ap89"
print('dataset:%s' % data)
folder = '/../../data/'+data+'/intermediate/'
start = time.time()
print('loading eid and name maps')
eid2ename, ename2eid = util.loadEidToEntityMap(curr_dir + folder+'entity2id.txt') #entity2id.txt
print('loading eid and skipgram maps')
eid2patterns, pattern2eids = util.loadFeaturesAndEidMap(curr_dir + folder+'eidSkipgramCounts.txt') #eidSkipgramCount.txt
print('loading skipgram strength map')
eidAndPattern2strength = util.loadWeightByEidAndFeatureMap(curr_dir + folder+'eidSkipgram2TFIDFStrength.txt', idx=-1) #(eid, feature, weight) file
if (FLAGS_USE_TYPE):
  print('loading eid and type maps')
  eid2types, type2eids = util.loadFeaturesAndEidMap(curr_dir + folder+'eidTypeCounts.txt') #eidTypeCount.txt
  print('loading type strength map')
  eidAndType2strength = util.loadWeightByEidAndFeatureMap(curr_dir + folder+'eidType2TFIDFStrength.txt', idx=-1) #(eid, feature, weight) file
end = time.time()
print("Finish loading all dataset, using %s seconds" % (end-start))

## Start set expansion

# read good_gold_set.json
fin = open(curr_dir + "/../../data/queries/good_gold_set.json")
gold_sets = json.load(fin)
fin.close()
for st in gold_sets:
  entities = list(gold_sets[st]["entities"].keys())
  print(entities)
  # 3 random samples for each set
  i = 1
  while i < 4: 
    userInput = random.sample(entities, 3)
    print("{} experiment {}: {}".format(st, i, str(userInput)))
    seedEidsWithConfidence = [(ename2eid[ele.lower().replace(' ', '_')], 0.0) for ele in userInput]

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
        FLAGS_VERBOSE=True,
        FLAGS_DEBUG=True
    )
    print("=== In test case ===")
    for ele in expandedEidsWithConfidence:
      print("eid=", ele[0], "ename=", eid2ename[ele[0]])

    if not os.path.exists(curr_dir + "/../../data/" + data + "/results/" + st):
      os.mkdir(curr_dir + "/../../data/" + data + "/results/" + st)
    with open(curr_dir + "/../../data/{}/results/{}/exp{}.txt".format(data, st, i), "w+") as fout:
      fout.write("Seeds: {}\n".format(str(userInput)))
      for ele in expandedEidsWithConfidence:
        fout.write("eid=" + str(ele[0]) + "\t" + "ename=" + eid2ename[ele[0]] + "\n")

    i += 1
