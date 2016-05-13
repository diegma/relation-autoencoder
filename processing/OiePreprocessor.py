__author__ = 'diego'

import argparse
import os
import sys
import time
from definitions import OieFeatures
from definitions import OieExample
print sys.path
import cPickle as pickle


class FeatureLexicon:

    def __init__(self):
        self.nextId = 0
        self.id2Str = {}
        self.str2Id = {}
        self.id2freq = {}
        self.nextIdPruned = 0
        self.id2StrPruned = {}
        self.str2IdPruned = {}

    def getOrAdd(self, s):
        if s not in self.str2Id:
            self.id2Str[self.nextId] = s
            self.str2Id[s] = self.nextId
            self.id2freq[self.nextId] = 1
            self.nextId += 1
        else:
            self.id2freq[self.str2Id[s]] += 1
        return self.str2Id[s]


    def getOrAddPruned(self, s):
        if s not in self.str2IdPruned:
            self.id2StrPruned[self.nextIdPruned] = s
            self.str2IdPruned[s] = self.nextIdPruned
            self.nextIdPruned += 1
        return self.str2IdPruned[s]

    def getId(self, s):
        if s not in self.str2Id:
            return None
        return self.str2Id[s]

    def getStr(self, idx):
        if idx not in self.id2Str:
            return None
        else:
            return self.id2Str[idx]

    def getStrPruned(self, idx):
        if idx not in self.id2StrPruned:
            return None
        else:
            return self.id2StrPruned[idx]

    def getFreq(self, idx):
        if idx not in self.id2freq:
            return None
        return self.id2freq[idx]


    def getDimensionality(self):
        return self.nextIdPruned
        # return self.nextId


def getFeatures(lexicon, featureExs, info, arg1=None, arg2=None, expand=False):
    feats = []
    for f in featureExs:
        res = f(info, arg1, arg2)
        if res is not None:
            if type(res) == list:
                for el in res:
                    featStrId = f.__name__ + "#" + el
                    if expand:
                        feats.append(lexicon.getOrAdd(featStrId))
                    else:
                        featId = lexicon.getId(featStrId)
                        if featId is not None:
                            feats.append(featId)
            else:
                featStrId = f.__name__ + "#" + res
                if expand:
                    feats.append(lexicon.getOrAdd(featStrId))
                else:
                    featId = lexicon.getId(featStrId)
                    if featId is not None:
                        feats.append(featId)

    return feats

def getFeaturesThreshold(lexicon, featureExs, info, arg1=None, arg2=None, expand=False, threshold=0):
    feats = []
    for f in featureExs:
        res = f(info, arg1, arg2)
        if res is not None:
            if type(res) == list:
                for el in res:
                    featStrId = f.__name__ + "#" + el
                    if expand:
                        if lexicon.id2freq[lexicon.getId(featStrId)] > threshold:
                            feats.append(lexicon.getOrAddPruned(featStrId))
                    else:
                        featId = lexicon.getId(featStrId)
                        if featId is not None:
                            if lexicon.id2freq[featId] > threshold:
                                feats.append(lexicon.getOrAddPruned(featStrId))
            else:
                featStrId = f.__name__ + "#" + res
                if expand:
                    if lexicon.id2freq[lexicon.getId(featStrId)] > threshold:
                        feats.append(lexicon.getOrAddPruned(featStrId))
                else:
                    featId = lexicon.getId(featStrId)
                    if featId is not None:
                        if lexicon.id2freq[featId] > threshold:
                            feats.append(lexicon.getOrAddPruned(featStrId))

    return feats

def prepareArgParser():
    parser = argparse.ArgumentParser(description='Processes an Oie file and add its representations '
                                                 'to a Python pickled file.')

    parser.add_argument('input_file', metavar='input-file',  help='input file in the Yao format')

    parser.add_argument('pickled_dataset', metavar='pickled-dataset', help='pickle file to be used to store output '
                                                                           '(created if empty)')

    parser.add_argument('--batch-name', default="train", nargs="?", help='name used as a reference in the pickled file')

    parser.add_argument('--features', default="basic", nargs="?", help='features (basic vs ?)')
    parser.add_argument('--threshold', default="0", nargs="?", type=int, help='minimum feature frequency')



    parser.add_argument('--test-mode', action='store_true',
                         help='used for test files '
                              '(the feature space is not expanded to include previously unseen features)')


    return parser

def loadExamples(fileName):
    count = 0
    with open(fileName, 'r') as fp:
        relationExamples = []
        for line in fp:
            line.strip()
            if len(line) == 0 or len(line.split()) == 0:
                raise IOError

            else:
                fields = line.split('\t')
                assert len(fields) == 9, "a problem with the file format (# fields is wrong) len is " \
                                         + str(len(fields)) + "instead of 9"
                # this will be 10
                relationExamples.append([str(count)] + fields)
                count += 1

    return relationExamples

# if __name__ == '__main__':
#     examples = loadExamples('/Users/admin/isti/amsterdam/data/candidate-100.txt')
#     print "Using basic features"
#     argFeatureExtrs = OieFeatures.getBasicFeatures()
#     ex = examples[0]
#     print ex
#     features = argFeatureExtrs
#
#     s = []
#     for f in features:
#         res = f([ex[1], ex[4], ex[5], ex[7]], ex[2], ex[3])
#         if res is not None:
#             s.append(f.__name__ + "#" + res)
#
#     print s, 'dd'

if __name__ == '__main__':

    tStart = time.time()

    print "Parameters: " + str(sys.argv[1::])
    parser = prepareArgParser()
    args = parser.parse_args()

    print "Parsed params: " + str(args)

    print "Loading sentences...",
    relationExamples = loadExamples(args.input_file)

    tEnd = time.time()
    print "Done (" + str(tEnd - tStart) + "s.)"

    # predFeatureExtrs = definitions.SrlFeatures.getJohanssonPredDisFeatures()
    #
    featureExtrs = None
    if args.features == "basic":
        print "Using rich features"
        featureExtrs = OieFeatures.getBasicCleanFeatures()

    relationLexicon = FeatureLexicon()

    dataset = {}
    goldstandard = {}

    if os.path.exists(args.pickled_dataset):
        tStart = time.time()
        print "Found existing pickled dataset, loading...",

        pklFile = open(args.pickled_dataset, 'rb')

        featureExtrs = pickle.load(pklFile)
        relationLexicon = pickle.load(pklFile)
        dataset = pickle.load(pklFile)
        goldstandard = pickle.load(pklFile)

        pklFile.close()
        tEnd = time.time()
        print "Done (" + str(tEnd - tStart) + "s.)"

    tStart = time.time()
    print "Processing relation Examples",

    examples = []
    relationLabels = {}
    if args.batch_name in dataset:
        examples = dataset[args.batch_name]
        relationLabels = goldstandard[args.batch_name]
    else:
        dataset[args.batch_name] = examples
        goldstandard[args.batch_name] = relationLabels

    reIdx = 0
    c = 0
    for re in relationExamples:
        getFeatures(relationLexicon, featureExtrs, [re[1], re[4], re[5], re[7], re[8], re[6]],
                                                             re[2], re[3], True)
    for re in relationExamples:
        reIdx += 1
        if reIdx % 1000 == 0:
            print ".",
        if reIdx % 10000 == 0:
            print reIdx,


        relationE = ''
        if re[9] != '':
            relationE = re[9]
        # print re[9]
        # if re[10] != '':
        #     if relationE != '':
        #         relationE += ' '+re[10]
        #     else:
        #         relationE = re[10]

        ex = OieExample.OieExample(re[2], re[3], getFeaturesThreshold(relationLexicon,
                                                             featureExtrs,
                                                             [re[1], re[4], re[5], re[7], re[8], re[6]],
                                                             # [re[1], re[4], re[5], re[7]],
                                                             re[2], re[3], True, threshold=args.threshold), re[5]
                                                             ,relation=relationE
                                   )
        relationLabels[c] = re[-1].strip().split(' ')
        c += 1

        examples.append(ex)


    tEnd = time.time()
    print "Done (" + str(tEnd - tStart) + "s.), processed " + str(len(examples))

    tStart = time.time()
    print "Pickling the dataset...",

    pklFile = open(args.pickled_dataset, 'wb')
    #pklFile = gzip.GzipFile(args.pickled_dataset, 'wb')

    pklProtocol = 2
    pickle.dump(featureExtrs, pklFile, protocol=pklProtocol)
    pickle.dump(relationLexicon, pklFile, protocol=pklProtocol)
    pickle.dump(dataset, pklFile, protocol=pklProtocol)
    pickle.dump(goldstandard, pklFile, protocol=pklProtocol)

    tEnd = time.time()
    print "Done (" + str(tEnd - tStart) + "s.)"