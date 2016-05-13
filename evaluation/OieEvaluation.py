__author__ = 'diego'

import pickle
import math
import argparse
import os
import sys
from processing.OiePreprocessor import FeatureLexicon
class multiLabelClusterEvaluation:
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

    def createResponse(self, response):
        self.numberOfElements, self.responseSets = self.createResponseSets(response)


    def b3precisionAvg(self, response_a, list_reference_a):
        prec_sum = 0.0
        denominator = float(len(self.assessableElemSet.intersection(response_a)))
        for ref in list_reference_a:
            prec_sum += len(response_a.intersection(ref)) / denominator
        return prec_sum / float(len(list_reference_a))

    def b3recallAvg(self, response_a, list_reference_a):
        rec_sum = 0.0
        for ref in list_reference_a:
            rec_sum += len(response_a.intersection(ref)) / float(len(ref))
        return rec_sum / float(len(list_reference_a))

    def b3precisionMax(self, response_a, list_reference_a):
        prec = []
        denominator = float(len(self.assessableElemSet.intersection(response_a)))
        for ref in list_reference_a:
            prec.append(len(response_a.intersection(ref)) / denominator)
        return max(prec)

    def b3recallMax(self, response_a, list_reference_a):
        rec = []
        for ref in list_reference_a:
            rec.append(len(response_a.intersection(ref)) / float(len(ref)))
        return max(rec)

    def b3TotalPrecision(self, avgOrMax):
        totalPrecision = 0.0
        for c in self.responseSets:
            for r in self.responseSets[c]:
                if r in self.assessableElemSet:
                    if avgOrMax == 'avg':
                        totalPrecision += self.b3precisionAvg(self.responseSets[c],
                                                              self.findCluster(r, self.referenceSets))
                    elif avgOrMax == 'max':
                        # print self.b3precisionMax(self.responseSets[c],
                        #                                       self.findCluster(r, self.referenceSets)),'g'
                        # print totalPrecision, len(self.assessableElemSet)

                        totalPrecision += self.b3precisionMax(self.responseSets[c],
                                                              self.findCluster(r, self.referenceSets))

        return totalPrecision / float(len(self.assessableElemSet))

    def b3TotalRecall(self, avgOrMax):
        totalRecall = 0.0
        for c in self.responseSets:
            for r in self.responseSets[c]:
                if r in self.assessableElemSet:
                    if avgOrMax == 'avg':
                        totalRecall += self.b3recallAvg(self.responseSets[c], self.findCluster(r, self.referenceSets))
                    elif avgOrMax == 'max':
                        totalRecall += self.b3recallMax(self.responseSets[c], self.findCluster(r, self.referenceSets))
        return totalRecall / float(len(self.assessableElemSet))



    # # UNiform cluster method
    # def b3TotalPrecisionUC(self, avgOrMax):
    #     totalPrecision = 0.0
    #     for c in self.responseSets:
    #         for r in self.responseSets[c]:
    #             if r in self.assessableElemSet:
    #                 clusterR = self.findCluster(r, self.referenceSets)
    #                 if avgOrMax == 'avg':
    #
    #                     totalPrecision += self.b3precisionAvg(self.responseSets[c],
    #                                                           clusterR) /
    #                 elif avgOrMax == 'max':
    #                     totalPrecision += self.b3precisionMax(self.responseSets[c],
    #                                                           clusterR)
    #
    #     return totalPrecision / len(self.assessableElemSet)
    #
    # # UNiform cluster method
    # def b3TotalRecallUC(self, avgOrMax):
    #     totalRecall = 0.0
    #     for c in self.responseSets:
    #         for r in self.responseSets[c]:
    #             if r in self.assessableElemSet:
    #                 if avgOrMax == 'avg':
    #                     totalRecall += self.b3recallAvg(self.responseSets[c], self.findCluster(r, self.referenceSets))
    #                 elif avgOrMax == 'max':
    #                     totalRecall += self.b3recallMax(self.responseSets[c], self.findCluster(r, self.referenceSets))
    #     return totalRecall / len(self.assessableElemSet)

    def createResponseSets(self, response):
        responseSets = {}
        numElem = 0
        for c in response:
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
            for category in relations[rel]:
                if category != '':
                    assessableElems.add(rel)
                    if category in referenceSets:
                        referenceSets[category].add(rel)
                    else:
                        referenceSets[category] = set([rel])
        return referenceSets, assessableElems

    def createValidationReferenceSets(self, referencePath, validationPath):
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
                    relations[c] = ''
                c += 1
        self.relations = relations
        referenceSets = {}
        assessableElems = set()
        for rel in relations:
            for category in relations[rel]:
                if category != '':
                    assessableElems.add(rel)
                    if category in referenceSets:
                        referenceSets[category].add(rel)
                    else:
                        referenceSets[category] = set([rel])
        return referenceSets, assessableElems

    def createReferenceSetsFromData(self, relations):
        self.relations = relations
        referenceSets = {}
        assessableElems = set()
        for rel in relations:
            for category in relations[rel]:
                if category != '':
                    # print 'category', category
                    assessableElems.add(rel)
                    if category in referenceSets:
                        referenceSets[category].add(rel)
                    else:
                        referenceSets[category] = set([rel])
        return referenceSets, assessableElems

    def findCluster(self, a, setsDictionary):
        foundClusters = []
        for c in setsDictionary:
            if a in setsDictionary[c]:
                foundClusters.append(setsDictionary[c])
        return foundClusters

    # def muc3Recall(self):
    #     numerator = 0.0
    #     denominator = 0.0
    #     for c in self.referenceSets:
    #         numerator += len(self.referenceSets[c]) - self.overlap(self.referenceSets[c], self.responseSets)
    #         denominator += len(self.referenceSets[c]) - 1
    #     if denominator == 0.0:
    #         return 0.0
    #     else:
    #         return numerator / denominator

    # def muc3Precision(self):
    #     numerator = 0.0
    #     denominator = 0.0
    #     for c in self.responseSets:
    #         print self.lenAssessableResponseCat(self.responseSets[c]), self.overlap(self.responseSets[c], self.referenceSets)
    #         numerator += self.lenAssessableResponseCat(self.responseSets[c]) - self.overlap(self.responseSets[c], self.referenceSets)
    #         lenRespo = self.lenAssessableResponseCat(self.responseSets[c])
    #         if lenRespo != 0:
    #             denominator += self.lenAssessableResponseCat(self.responseSets[c]) - 1
    #     if denominator == 0.0:
    #         return 0.0
    #     else:
    #         return numerator / denominator

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


        recAvg = self.b3TotalRecall('avg')
        recMax = self.b3TotalRecall('max')
        precAvg = self.b3TotalPrecision('avg')
        precMax = self.b3TotalPrecision('max')

        if recAvg == 0 and precAvg == 0:
            F1Avg = 0.0
            F05Avg = 0.0
        else:
            betasquare = math.pow(0.5, 2)
            F1Avg = (2 * recAvg * precAvg) / (recAvg + precAvg)
            F05Avg = ((1+betasquare) * recAvg * precAvg)/((betasquare*precAvg)+recAvg)

        if recMax == 0 and precMax == 0:
            F1Max = 0
            F05Max = 0
        else:
            F1Max = (2 * recMax * precMax) / (recMax + precMax)
            betasquare = math.pow(0.5, 2)
            F05Max = ((1+betasquare) * recMax * precMax)/((betasquare*precMax)+recMax)

        print validOrTrain, 'B3 F1 Avg =', F1Avg, 'F0.5Avg =', F05Avg, 'B3 recall Avg =', recAvg, 'B3 precision Avg =', precAvg

        print validOrTrain, 'B3 F1 Max =', F1Max, 'F0.5Max =', F05Max, 'B3 recall Max =', recMax, 'B3 precision Max =', precMax

        # print 'muc recall =', self.muc3Recall(), 'muc precision =', self.muc3Precision()


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
            # if relations[rel][0] != '':
            if relations[rel][0] == '/location/location/containedby':  # this is used to assess the performance on containedby
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


        recB3c = self.b3TotalClusterRecall()
        precB3c = self.b3TotalClusterPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3c == 0.0 and precB3c == 0.0:
            F1B3c = 0.0
            F05B3c = 0.0
        else:
            betasquare = math.pow(0.5, 2)
            F1B3c = (2 * recB3c * precB3c) / (recB3c + precB3c)
            F05B3c = ((1+betasquare) * recB3c * precB3c)/((betasquare*precB3c)+recB3c)

        print validOrTrain, ' Clusterwise B3 F1 =', F1B3c, 'F0.5 =', F05B3c, 'B3 recall =', recB3c, 'B3 precision =', precB3c


        recPW = self.muc3Recall()
        precPW = self.muc3Precision()
        if recPW == 0.0 and precPW == 0.0:
            F1PW = 0.0
            F05PW = 0.0
        else:
            F1PW = (2 * recPW * precPW) / (recPW + precPW)
            F05PW = ((1+betasquare) * recPW * precPW)/((betasquare*precPW)+recPW)
        print validOrTrain, 'Pairwise F1 =', F1PW, 'F0.5 =', F05PW, 'Pairwise recall  =', recPW, 'Pairwise precision =', precPW

        # print 'muc recall =', self.muc3Recall(), 'muc precision =', self.muc3Precision()

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

def evaluateLDA():
        clusters = pickle.load(open('/Users/admin/Downloads/I_TTVI_TTVCompleteBasicSingle.pkl_lr0.1_ns20_Init-E-3+E-3batch100Emb30_rank5_optimization1_ltwo0.1_EX_EMBWordsTrue_freq_evalFalse_maxepoch5_lr0.1_embedsize30_l10.0_l20.1_opt1_rank5_rel_num100_batch100_negs20_epoch5_test', 'rb'))
        path = '/Users/enfry/work/amsterdam/data/'
        # evaluationTest = singleLabelClusterEvaluation(path+'candidate-2000s.context.filtered.triples.pathfiltered.single-relation.sortedondate.RelLDA2012.txt',
        #                                               True,
        #                                               validationPath=path+'candidate-2000s.context.filtered.triples.pathfiltered.single-relation.sortedondate.test.80%.RelLDA2012.txt')
        # evaluationValid = singleLabelClusterEvaluation(path+'candidate-2000s.context.filtered.triples.pathfiltered.single-relation.sortedondate.RelLDA2012.txt', True, validationPath=path+'validation/candidate-2000s.context.filtered.triples.pathfiltered.single-relation.sortedondate.validation.20%.RelLDA2012.txt')
        evaluationTest = singleLabelClusterEvaluation(path+'candidate-2000s.context.filtered.triples.pathfiltered.single-relation.sortedondate.test.80%.RelLDA2012.txt',
                                                      True,
                                                      validationPath='')

        evaluationTest.createResponse(clusters)
        evaluationTest.printEvaluation('Test')

        # evaluationValid.createResponse(clusters)
        # evaluationValid.printEvaluation('Validation')

if __name__ == '__main__':
    # response = {}
    # reference = {}
    #
    # response[0] = set([1, 2, 3, 4, 5, 8, 9, 10, 11, 12])
    # response[10] = set([6, 7])
    # # response[1] = set(['b','g'])
    # # response[2] = set(['ee', 'ff'])
    #
    # reference[1] = ['A']
    # reference[2] = ['A']
    # reference[3] = ['A']
    # reference[4] = ['A']
    # reference[5] = ['A']
    # reference[6] = ['B']
    # reference[7] = ['B']
    # reference[8] = ['C']
    # reference[9] = ['C']
    # reference[10] = ['C']
    # reference[11] = ['C']
    # reference[12] = ['C']
    #
    # eval = singleLabelClusterEvaluation(reference, False)
    # eval.createResponse(response)
    # eval.printEvaluation('Training')

    # evaluateLDA()

    args = getCommandArgs()
    goldStandard = loadData(args.pickled_dataset)
    results = pickle.load(open(args.pickled_results, 'rb'))
    if args.pickled_dataset.find('Single') > 0:
        single_eval = singleLabelClusterEvaluation(goldStandard['test'], False)
        single_eval.createResponse(results)
        single_eval.printEvaluation('Test')
    else:
        multi_eval = multiLabelClusterEvaluation(goldStandard['train'], False)
        multi_eval.createResponse(results)
        multi_eval.printEvaluation('Training')