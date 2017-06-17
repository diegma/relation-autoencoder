__author__ = 'diego'


import math as m
import numpy as np
import scipy.sparse as sp
import theano
from definitions import settings
import cPickle as pickle

class MatrixDataSet:
    # matrix formatted dataset
    def __init__(self, arguments1, arguments2, argFeatures, negArgs1, negArgs2):
        self.args1 = arguments1  # (l)
        self.args2 = arguments2  # (l)
        self.xFeats = argFeatures  # (l, h)
        self.neg1 = negArgs1  # (n, l)
        self.neg2 = negArgs2  # (n, l)


class MatrixDataSetNoEncoding:
    # matrix formatted dataset
    def __init__(self, arguments1, arguments2, realProbs):
        self.args1 = arguments1  # (l)
        self.args2 = arguments2  # (l)
        self.realProbs = realProbs  # (l, r)





class DataSetManager:
    """
    Attributes:
        'id2Arg', 
        'negSamplesNum', int: neg sample number
        'negSamplingCum', list of float: 
        'negSamplingDistr', list of float:
        'negSamplingDistrPower', bool:
        'validExs',  
        'featureLex', OiePreprocessor.FeatureLexicon
        'rng', : random state generator
        'arg2Id', dict: entity string to id
        'testExs', OieData.MatrixDataSet
        'relationNum', int: relation number
        'trainExs',OieData.MatrixDataSet:
    """
    def __init__(self, oieDataset, featureLex, rng, negSamplesNum, relationNum, negSamplingDistrPower=0.75):

        self.negSamplesNum = negSamplesNum  # the number of negative samples considered

        self.negSamplingDistrPower = negSamplingDistrPower  # the sampling distribution for negative sampling

        self.rng = rng # random number generator

        self.relationNum = relationNum  # relation number

        # id2Str, str2Id
        self.featureLex = featureLex  # feature lexicon

        # sets id2Arg1, id2Arg2, arg12Id, arg22Id, neg1SamplingDistr, neg2SamplingDistr
        self._extractArgsMappings(oieDataset)

        # each examples csr_matrix[exampleNum x getDimensionality()], labels are numpy.array


        # self.validExs = self._extractExamples(oieDataset['dev'])

        self.trainExs = self._extractExamples(oieDataset['train'])
        if 'dev' in oieDataset:
            self.validExs = self._extractExamples(oieDataset['dev'])
        else:
            self.validExs = None

        if 'test' in oieDataset:
            self.testExs = self._extractExamples(oieDataset["test"])
        else:
            self.testExs = None

    def _sample(self, cutoffs):
        idx = cutoffs.searchsorted(self.rng.uniform(0, cutoffs[-1]))
        return idx


    def _sample1(self, distr):

        # check numpy, it should have some efficient ways to sample from multinomials
        val = self.rng.uniform()
        pos = 0
        for idx in xrange(len(distr)):
            pos += distr[idx]
            if pos > val:
                return idx
        return len(distr) - 1


    def _extractExamples(self, oieExamples):

        l = len(oieExamples)
        n = self.negSamplesNum

        args1 = np.zeros(l, dtype=np.int32)  # entity1 ids
        args2 = np.zeros(l, dtype=np.int32)  # entity2 ids


        neg1 = np.zeros((n, l), dtype=np.int32)  # negative samples for the dataset: is it fixed? or change among different iteration?
        neg2 = np.zeros((n, l), dtype=np.int32)  #


        # print self.featureLex.getDimensionality()
        # l * feature_dim 0-1 matrix.
        xFeatsDok = sp.dok_matrix((l, self.featureLex.getDimensionality()), dtype=theano.config.floatX)
                                                                          #  @UndefinedVariable float32

        for i, oieEx in enumerate(oieExamples):
            args1[i] = self.arg2Id[oieEx.arg1]
            args2[i] = self.arg2Id[oieEx.arg2]

            for feat in oieEx.features:
                xFeatsDok[i, feat] = 1

            # should do it differently (sample random indexes during training), see below TODO

            for k in xrange(n):
                neg1[k, i] = self._sample(self.negSamplingCum)  # np.searchsorted

            for k in xrange(n):
                neg2[k, i] = self._sample(self.negSamplingCum)
            


        xFeats = sp.csr_matrix(xFeatsDok, dtype="float32")

        return MatrixDataSet(args1, args2, xFeats, neg1, neg2)

    def _indexElements(self, elements):

        idx = 0
        id2Elem = {}
        elem2Id = {}
        for x in elements:
            id2Elem[idx] = x
            elem2Id[x] = idx
            idx += 1
        return id2Elem, elem2Id

    def _extractArgsMappings(self, oieDataset):

        # sets id2Arg1, id2Arg2, arg12Id, arg22Id, neg1SamplingDistr, neg2SamplingDistr
        argFreqs = {} # frequency of unique entity
        for key in oieDataset:  # key: 'train', 'test' ... Todo: is he count test set?
            for oieEx in oieDataset[key]:  # here it iterates over train, test, dev. oieEx: OieExample
                if oieEx.arg1 not in argFreqs:
                    argFreqs[oieEx.arg1] = 1
                else:
                    argFreqs[oieEx.arg1] += 1

                if oieEx.arg2 not in argFreqs:
                    argFreqs[oieEx.arg2] = 1
                else:
                    argFreqs[oieEx.arg2] += 1



        self.id2Arg, self.arg2Id = self._indexElements(argFreqs)

        # Todo: can rewrite by numpy
        argSampFreqs = [float(argFreqs[self.id2Arg[val]]) for val in xrange(len(self.id2Arg))]  # id to freq, list of float
        argSampFreqsPowered = map(lambda x: m.pow(x, self.negSamplingDistrPower),  argSampFreqs)  # seems same as mikolove
        norm1 = reduce(lambda x, y: x + y,  argSampFreqsPowered)  # sum of argSampFreqsPowered
        self.negSamplingDistr = map(lambda x: x / norm1, argSampFreqsPowered)  # divide norm, the probability for each entity id
        self.negSamplingCum = np.cumsum(self.negSamplingDistr)  # like the stick beark position for calculating probability.




    def getArgVocSize(self):
        return len(self.arg2Id)


    def getDimensionality(self):
        return self.featureLex.getDimensionality()

    def getNegNum(self):
        return self.negSamplesNum

    def getTrainSet(self):
        return self.trainExs

    def getValidSet(self):
        return self.validExs

    def getTestSet(self):
        return self.testExs

    def getRelationNum(self):
        return self.relationNum

    def getExampleFeatures(self, id):
        a = []
        for e in self.trainExs.xFeats[id].nonzero()[1]:
            feat = self.featureLex.getStrPruned(e)
            if (self.featureLex.getStrPruned(e).find('trigger') > -1 or
                self.featureLex.getStrPruned(e).find('arg1') > -1 or
                self.featureLex.getStrPruned(e).find('arg2') > -1):
                a.append(feat)
            # else:  # only for debugging purposes, should be commented
            #     a.append(feat)
        return a

    def getExampleFeature(self, id, feature):
        for e in self.trainExs.xFeats[id].nonzero()[1]:
            feat = self.featureLex.getStrPruned(e)
            if self.featureLex.getStrPruned(e).find(feature) > -1:
                return feat
        return None

    def getExampleFeatureValid(self, id, feature):
        for e in self.validExs.xFeats[id].nonzero()[1]:
            feat = self.featureLex.getStrPruned(e)
            if self.featureLex.getStrPruned(e).find(feature) > -1:
                return feat
        return None

    def getExampleFeatureTest(self, id, feature):
        for e in self.testExs.xFeats[id].nonzero()[1]:
            feat = self.featureLex.getStrPruned(e)
            if self.featureLex.getStrPruned(e).find(feature) > -1:
                return feat
        return None

    def getNegSamplingCum(self):
        return self.negSamplingCum



