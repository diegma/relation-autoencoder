__author__ = 'diego'

import pickle
import math
import argparse
import os
import sys
from processing.OiePreprocessor import FeatureLexicon

class singleLabelClusterEvaluation:
    def __init__(self, referencePath, file, validationPath=''):
        self.relations = {}
        if file:
            if validationPath != '':
                self.referenceSets, self.assessableElemSet = self.createValidationReferenceSets(referencePath,
                                                                                                validationPath)
            else:
                self.referenceSets, self.assessableElemSet = self.createReferenceSets(referencePath)

        else:
            self.referenceSets, self.assessableElemSet = self.createReferenceSetsFromData(referencePath)
            # print self.referenceSets
            # print self.assessableElemSet

    def createResponse(self, response):
        self.numberOfElements, self.responseSets = self.createResponseSets(response)
        # print self.responseSets



    def b3precision(self, response_a, reference_a):
        # print response_a.intersection(self.assessableElemSet), 'in precision'
        return len(response_a.intersection(reference_a)) / float(len(response_a.intersection(self.assessableElemSet)))

    def b3recall(self, response_a, reference_a):
        return len(response_a.intersection(reference_a)) / float(len(reference_a))



    def b3TotalElementPrecision(self):
        totalPrecision = 0.0
        for c in self.responseSets:
            for r in self.responseSets[c]:
                if r in self.assessableElemSet:
                    # print r
                    totalPrecision += self.b3precision(self.responseSets[c],
                                                       self.findCluster(r, self.referenceSets))

        return totalPrecision / float(len(self.assessableElemSet))

    def b3TotalElementRecall(self):
        totalRecall = 0.0
        for c in self.responseSets:
            for r in self.responseSets[c]:
                if r in self.assessableElemSet:
                    totalRecall += self.b3recall(self.responseSets[c], self.findCluster(r, self.referenceSets))

        return totalRecall / float(len(self.assessableElemSet))


    def b3TotalClusterPrecision(self):
        totalPrecision = 0.0
        for c in self.responseSets:
            for r in self.responseSets[c]:
                if r in self.assessableElemSet:
                    totalPrecision += self.b3precision(self.responseSets[c],
                                                       self.findCluster(r, self.referenceSets)) / \
                                      float(len(self.responseSets)*len(self.responseSets[c]))
        return totalPrecision

    def b3TotalClusterRecall(self):
        totalRecall = 0.0
        for c in self.responseSets:
            for r in self.responseSets[c]:
                if r in self.assessableElemSet:
                    totalRecall += self.b3recall(self.responseSets[c], self.findCluster(r, self.referenceSets)) / \
                                   float(len(self.responseSets)*len(self.responseSets[c]))

        return totalRecall


    def createResponseSets(self, response):
        responseSets = {}
        numElem = 0
        for c in response:
            if len(response[c]) > 0:
                numElem += len(response[c])
                responseSets[c] = set(response[c])

        return numElem, responseSets



    def createReferenceSets(self, referencePath):
        with open(referencePath, 'r') as f:
            relations = {}
            c = 0
            for line in f:
                lineSplit = line.split('\t')
                relations[c] = lineSplit[-1].strip().split(' ')
                c += 1
        self.relations = relations
        referenceSets = {}
        assessableElems = set()
        for rel in relations:
            if relations[rel][0] != '':
                assessableElems.add(rel)
                if relations[rel][0] in referenceSets:
                    referenceSets[relations[rel][0]].add(rel)
                else:
                    referenceSets[relations[rel][0]] = set([rel])
        return referenceSets, assessableElems

    def createValidationReferenceSets(self, referencePath, validationPath):
        # referencePath is usually the entire training set
        with open(referencePath, 'r') as f, open(validationPath, 'r') as f1:
            validationSet = {}
            for line in f1:
                if line not in validationSet:
                    validationSet[line] = 1

            relations = {}
            c = 0
            for line in f:
                if line in validationSet:
                    lineSplit = line.split('\t')
                    relations[c] = lineSplit[-1].strip().split(' ')
                else:
                    relations[c] = ['']
                c += 1
        # self.relationsValid = relations
        referenceSets = {}
        assessableElems = set()
        for rel in relations:
            if relations[rel][0] != '':
                assessableElems.add(rel)
                if relations[rel][0] in referenceSets:
                    referenceSets[relations[rel][0]].add(rel)
                else:
                    referenceSets[relations[rel][0]] = set([rel])
        return referenceSets, assessableElems


    def createReferenceSetsFromData(self, relations):
        self.relations = relations
        referenceSets = {}
        assessableElems = set()
        for rel in relations:
            if relations[rel][0] != '':
                # print 'category', category
                assessableElems.add(rel)
                if relations[rel][0] in referenceSets:
                    referenceSets[relations[rel][0]].add(rel)
                else:
                    referenceSets[relations[rel][0]] = set([rel])
        return referenceSets, assessableElems

    def findCluster(self, a, setsDictionary):
        foundClusters = []
        for c in setsDictionary:
            if a in setsDictionary[c]:
                return setsDictionary[c]
        #         foundClusters.append(setsDictionary[c])
        # return foundClusters

    def muc3Recall(self):
        numerator = 0.0
        denominator = 0.0
        for c in self.referenceSets:
            numerator += len(self.referenceSets[c]) - self.overlap(self.referenceSets[c], self.responseSets)
            denominator += len(self.referenceSets[c]) - 1
        if denominator == 0.0:
            return 0.0
        else:
            return numerator / denominator

    def muc3Precision(self):
        numerator = 0.0
        denominator = 0.0
        for c in self.responseSets:
            if len(self.responseSets[c]) > 0:
                # print self.lenAssessableResponseCat(self.responseSets[c]), self.overlap(self.responseSets[c], self.referenceSets)
                numerator += self.lenAssessableResponseCat(self.responseSets[c]) - self.overlap(self.responseSets[c], self.referenceSets)
                lenRespo = self.lenAssessableResponseCat(self.responseSets[c])
                if lenRespo != 0:
                    denominator += self.lenAssessableResponseCat(self.responseSets[c]) - 1
        if denominator == 0.0:
            return 0.0
        else:
            return numerator / denominator

    def overlap(self, a, setsDictionary):
        numberIntersections = 0
        for c in setsDictionary:
            if len(a.intersection(setsDictionary[c])) > 0:
                numberIntersections += 1
        return numberIntersections


    def lenAssessableResponseCat(self, responesSet_c):
        length = 0
        for r in responesSet_c:
            if r in self.assessableElemSet:
                length += 1
        return length

    def printEvaluation(self, validOrTrain):


        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
            F05B3 = 0.0
        else:
            betasquare = math.pow(0.5, 2)
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
            F05B3 = ((1+betasquare) * recB3 * precB3)/((betasquare*precB3)+recB3)

        print validOrTrain, ' Elementwise B3 F1 =', F1B3, 'F0.5 =', F05B3, 'B3 recall =', recB3, 'B3 precision =', precB3


        

    def getF05(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F05B3 = 0.0
        else:
            F05B3 = ((1+betasquare) * recB3 * precB3)/((betasquare*precB3)+recB3)
        return F05B3

    def getF1(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()

        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
        else:
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
        return F1B3

def loadData(pickled_dataset):

    if not os.path.exists(pickled_dataset):
        print "Pickled dataset not found"
        sys.exit()

    pklFile = open(pickled_dataset, 'rb')

    featureExtrs = pickle.load(pklFile)

    relationLexicon = pickle.load(pklFile)

    data = pickle.load(pklFile)

    goldStandard = pickle.load(pklFile)

    pklFile.close()


    return goldStandard

def getCommandArgs():
    parser = argparse.ArgumentParser(description='Trains a basic Open Information Extraction Model')

    parser.add_argument('--pickled_dataset', metavar='pickled_dataset', nargs='?', required=True,
                        help='the pickled dataset file (produced by OiePreprocessor.py)')
    parser.add_argument('--pickled_results', metavar='pickled_results', nargs='?', required=True,
                        help='the pickled results file (produced by OiePreprocessor.py)')


    return parser.parse_args()


