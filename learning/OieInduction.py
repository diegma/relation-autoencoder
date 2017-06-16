__author__ = 'diego'

import argparse
import os

import numpy as np

import sys
import time
import cPickle as pickle
import operator
from theano import sparse
import theano
import theano.tensor as T
from learning.OieModel import OieModelFunctions

from learning.OieData import DataSetManager
from learning.OieData import MatrixDataSet
from processing.OiePreprocessor import FeatureLexicon
from evaluation.OieEvaluation import singleLabelClusterEvaluation
import definitions.settings as settings
from learning.NegativeExampleGenerator import NegativeExampleGenerator
from collections import OrderedDict

class ReconstructInducer(object):

    def __init__(self, data, goldStandard, rand, epochNum, learningRate, batchSize, embedSize, lambdaL1, lambdaL2,
                 optimization, modelName, model, fixedSampling, extEmb, extendedReg,
                 frequentEval, alpha):
        self.rand = rand
        self.data = data
        self.goldStandard = goldStandard
        self.optimization = optimization
        self.modelName = modelName
        self.model = model
        self.relationNum = data.getRelationNum()
        self.extEmb = extEmb
        self.extendedReg = extendedReg
        self.frequentEval = frequentEval
        self.alpha = alpha

        self.modelID = model + '_' + modelName+'_maxepoch'+str(epochNum)+'_lr'+str(learningRate)\
                        + '_embedsize' + str(embedSize) + '_l1' + str(lambdaL1) + '_l2' + str(lambdaL2) \
                        + '_opt' + str(optimization) + '_rel_num' + str(self.relationNum)+ \
                       '_batch' + str(batchSize) + '_negs' + str(data.negSamplesNum)

        self.modelFunc = OieModelFunctions(rand, data.getDimensionality(), embedSize,  self.relationNum,
                                           data.getArgVocSize(), model, self.data, self.extEmb, self.extendedReg,
                                           self.alpha)

        self.embedSize = embedSize
        self.epochNum = epochNum
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.lambdaL1 = lambdaL1
        self.lambdaL2 = lambdaL2
        self.fixedSampling = fixedSampling
        self.negativeSampler = NegativeExampleGenerator(rand, data.getNegSamplingCum())
        self.accumulator = []



    def _makeShared(self, matrixDataset, borrow=True):

        sharedMatrix = MatrixDataSet(
                arguments1=theano.shared(matrixDataset.args1, borrow=borrow),
                arguments2=theano.shared(matrixDataset.args2, borrow=borrow),
                argFeatures=theano.shared(matrixDataset.xFeats, borrow=borrow),
                negArgs1=theano.shared(matrixDataset.neg1, borrow=borrow),
                negArgs2=theano.shared(matrixDataset.neg2, borrow=borrow)
        )
        return sharedMatrix


    def compileFunction(self, learningRate, epochNum, batchSize, lambda1, lambda2):

        trainDataNP = self.data.getTrainSet()
        trainData = self._makeShared(trainDataNP)

        validDataNP = self.data.getValidSet()

        testDataNP = self.data.getTestSet()

        if validDataNP is not None:
            validData = self._makeShared(validDataNP)

        if testDataNP is not None:
            testData = self._makeShared(testDataNP)

        # build the symbolic computation

        batchIdx = T.lscalar()  # index to a [mini]batch
        xFeats = sparse.csr_matrix(name='x', dtype='float32')  # l, h

        args1 = T.ivector()  # l
        args2 = T.ivector()  # l
        neg1 = T.imatrix()  # n, l
        neg2 = T.imatrix()  # n, l

        print "Starting to build train err computation (not compiling it yet)"
        adjust = float(batchSize) / float(trainDataNP.args1.shape[0])

        cost = self.modelFunc.buildTrainErrComputation(batchSize, self.data.getNegNum(),
                                                           xFeats, args1, args2, neg1, neg2) + \
                       (lambda1 * self.modelFunc.L1 * adjust) + \
                       (lambda2 * self.modelFunc.L2 * adjust)

        if self.optimization == 1:
            from learning.Optimizers import AdaGrad
            ada = AdaGrad(self.modelFunc.params)
            updates = ada.update(self.learningRate, self.modelFunc.params, cost)
            if False:
                adaEncoder = AdaGrad(self.modelFunc.relationClassifiers.params)
                updatesEncoder = adaEncoder.update(self.learningRate, self.modelFunc.relationClassifiers.params, cost)

                adaDecoder = AdaGrad(self.modelFunc.argProjector.params)
                updatesDecoder = adaDecoder.update(self.learningRate, self.modelFunc.argProjector.params, cost)

        elif self.optimization == 0:
            from learning.Optimizers import SGD
            sgd = SGD()
            updates = sgd.update(self.learningRate, self.modelFunc.params, cost)



        print "Compiling train function..."



        trainModel = theano.function(inputs=[batchIdx, neg1, neg2],
                                     outputs=cost,
                                     updates=updates,
                                     givens={
                xFeats: trainData.xFeats[batchIdx * batchSize: (batchIdx + 1) * batchSize],
                args1: trainData.args1[batchIdx * batchSize: (batchIdx + 1) * batchSize],
                args2: trainData.args2[batchIdx * batchSize: (batchIdx + 1) * batchSize]
                                     }
            )
        if False:
            trainEncoder = theano.function(inputs=[batchIdx, neg1, neg2],
                                     outputs=cost,
                                     updates=updatesEncoder,
                                     givens={
                xFeats: trainData.xFeats[batchIdx * batchSize: (batchIdx + 1) * batchSize],
                args1: trainData.args1[batchIdx * batchSize: (batchIdx + 1) * batchSize],
                args2: trainData.args2[batchIdx * batchSize: (batchIdx + 1) * batchSize]
                                     }
            )
            trainDecoder = theano.function(inputs=[batchIdx, neg1, neg2],
                                     outputs=cost,
                                     updates=updatesDecoder,
                                     givens={
                xFeats: trainData.xFeats[batchIdx * batchSize: (batchIdx + 1) * batchSize],
                args1: trainData.args1[batchIdx * batchSize: (batchIdx + 1) * batchSize],
                args2: trainData.args2[batchIdx * batchSize: (batchIdx + 1) * batchSize]
                                     }
            )

        prediction = self.modelFunc.buildLabelComputation(batchSize, xFeats)

        print "Compiling label function (for training)..."
        labelTrain = theano.function(inputs=[batchIdx],
                                     outputs=prediction,
                                     updates=[],
                                     givens={
                xFeats: trainData.xFeats[batchIdx * batchSize:(batchIdx + 1) * batchSize]})

        if validDataNP is not None:
            print "Compiling label function (for validation)..."
            labelValid = theano.function(inputs=[batchIdx],
                                         outputs=prediction,
                                         updates=[],
                                         givens={xFeats: validData.xFeats[batchIdx * batchSize:
                                         (batchIdx + 1) * batchSize]})
        if testDataNP is not None:
            print "Compiling label function (for test)..."
            labelTest = theano.function(inputs=[batchIdx],
                                         outputs=prediction,
                                         updates=[],
                                         givens={xFeats: testData.xFeats[batchIdx * batchSize:
                                         (batchIdx + 1) * batchSize]})


        print "Done with compiling function."
        if validDataNP is not None and testDataNP is not None:

            return trainModel, labelTest, labelValid
        else:
            if False:
                return trainEncoder, trainDecoder, labelTrain
            else:
                return trainModel, labelTrain

    def learn(self):
        trainDataNP = self.data.getTrainSet()
        validDataNP = self.data.getValidSet()
        testDataNP = self.data.getTestSet()

        print "Starting to compile functions"


        if validDataNP is not None and testDataNP is not None:
            trainModel, labelTest, labelValid = self.compileFunction(self.learningRate, self.epochNum,
                                                                 self.batchSize, self.lambdaL1, self.lambdaL2)
        else:
            if False:
                trainEncoder, trainDecoder, labelTrain = self.compileFunction(self.learningRate, self.epochNum,
                                                      self.batchSize, self.lambdaL1, self.lambdaL2)
            else:
                trainModel, labelTrain = self.compileFunction(self.learningRate, self.epochNum,
                                                      self.batchSize, self.lambdaL1, self.lambdaL2)


        ###############
        # TRAIN MODEL #
        ###############

        # compute number of minibatches for training, validation and testing
        trainBatchNum = trainDataNP.args1.shape[0] / self.batchSize

        if validDataNP is not None and testDataNP is not None:
            validBatchNum = validDataNP.args1.shape[0] / self.batchSize
            validEval = singleLabelClusterEvaluation(self.goldStandard['dev'], False)

            testBatchNum = testDataNP.args1.shape[0] / self.batchSize
            testEval = singleLabelClusterEvaluation(self.goldStandard['test'], False)
        else:
            trainEval = singleLabelClusterEvaluation(self.goldStandard['train'], False)

        print str(trainBatchNum * self.batchSize) + " training examples, "
        # print trainDataNP.args1.shape[0], self.batchSize, trainBatchNum
        print '... training the model'
        startTime = time.clock()

        doneLooping = False
        epoch = 0


        while (epoch < self.epochNum) and (not doneLooping):
            negativeSamples1 = self.negativeSampler.generate_random_negative_example(trainDataNP.args1,
                                                                                     self.data.getNegNum())
            negativeSamples2 = self.negativeSampler.generate_random_negative_example(trainDataNP.args2,
                                                                                     self.data.getNegNum())

            err = 0
            epochStartTime = time.clock()

            epoch += 1
            print '\nEPOCH ' + str(epoch)
            for idx in xrange(trainBatchNum):
                if not self.fixedSampling:
                    neg1 = negativeSamples1[:, idx * self.batchSize: (idx + 1) * self.batchSize]
                    neg2 = negativeSamples2[:, idx * self.batchSize: (idx + 1) * self.batchSize]
                else:
                    neg1 = trainDataNP.neg1[:, idx * self.batchSize: (idx + 1) * self.batchSize]
                    neg2 = trainDataNP.neg2[:, idx * self.batchSize: (idx + 1) * self.batchSize]


                ls = trainModel(idx, neg1, neg2)
                err += ls

                # self.modelFunc.argProjector.normalize()
                # print('.'),
                if self.frequentEval:
                    if validDataNP is not None and testDataNP is not None:
                        if idx % 1 == 0:
                            print(str(idx * batchSize)),
                            print idx, '############################################################'
                            validCluster = self.getClustersSets(labelValid, validBatchNum)
                            validEval.createResponse(validCluster)
                            validEval.printEvaluation('Validation')

                            testCluster = self.getClustersSets(labelTest, testBatchNum)
                            testEval.createResponse(testCluster)
                            testEval.printEvaluation('Test')
                    else:
                        print(str(idx * batchSize)),
                        print idx, '############################################################'
                        trainClusters = self.getClustersPopulation(labelTrain, trainBatchNum)
                        print trainClusters
                        print


            epochEndTime = time.clock()

            print 'Training error ', str(err)
            print "Epoch time = " + str(epochEndTime - epochStartTime)

            if validDataNP is None or testDataNP is None:
                print 'Training Set'
                # print labelTrain(1)[1]
                trainClusters = self.getClustersSets(labelTrain, trainBatchNum)
                posteriorsTrain = [labelTrain(i)[1] for i in xrange(trainBatchNum)]
                trainPosteriors = [item for sublist in posteriorsTrain for item in sublist]
                # for p, probs in enumerate(predictions):
                #     print p, probs
                trainEval.createResponse(trainClusters)
                if self.modelName != 'Test':
                    trainEval.printEvaluation('Training')

                if self.modelName == 'Test':
                    self.getClustersWithFrequencies(trainClusters, self.data, settings.elems_to_visualize)
                else:
                    getClustersWithFrequencies(trainClusters, self.data, settings.elems_to_visualize)
                if not settings.debug:
                    pickleClustering(trainClusters, self.modelID+'_epoch'+str(epoch))
                    if epoch % 5 == 0 and epoch > 0:
                        picklePosteriors(trainPosteriors, self.modelID+'_Posteriors_epoch'+str(epoch))

            if validDataNP is not None and testDataNP is not None:

                validCluster = self.getClustersSets(labelValid, validBatchNum)
                posteriorsValid = [labelValid(i)[1] for i in xrange(validBatchNum)]
                validPosteriors = [item for sublist in posteriorsValid for item in sublist]
                validEval.createResponse(validCluster)
                validEval.printEvaluation('Validation')
                getClustersWithFrequenciesValid(validCluster, self.data, settings.elems_to_visualize)
                if not settings.debug:
                    pickleClustering(validCluster, self.modelID+'_epoch'+str(epoch)+'_valid')
                    if epoch % 5 == 0 and epoch > 0:
                        picklePosteriors(validPosteriors, self.modelID+'_Posteriors_epoch'+str(epoch)+'_valid')

                testCluster = self.getClustersSets(labelTest, testBatchNum)
                posteriorsTest = [labelTest(i)[1] for i in xrange(testBatchNum)]
                testPosteriors = [item for sublist in posteriorsTest for item in sublist]
                testEval.createResponse(testCluster)
                testEval.printEvaluation('Test')
                getClustersWithFrequenciesTest(testCluster, self.data, settings.elems_to_visualize)
                if not settings.debug:
                    pickleClustering(testCluster, self.modelID+'_epoch'+str(epoch)+'_test')
                    if epoch % 5 == 0 and epoch > 0:
                        picklePosteriors(testPosteriors, self.modelID+'_Posteriors_epoch'+str(epoch)+'_test')


        endTime = time.clock()
        print 'Optimization complete'
        print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (endTime - startTime))
        print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                              ' ran for %.1fs' % ((endTime - startTime)))




    def getClustersSets(self, labelTrain, trainBatchNum):
        clusters = {}
        for i in xrange(self.relationNum):
            clusters[i] = set()
        predictionsTrain = [labelTrain(i)[0] for i in xrange(trainBatchNum)]
        predictions = [item for sublist in predictionsTrain for item in sublist]  # returns the flatten() list
        for j in xrange(len(predictions)):
            clusters[predictions[j]].add(j)
        return clusters

    def getClustersPopulation(self, labelTrain, trainBatchNum):
        clusters = {}
        for i in xrange(self.relationNum):
            clusters[i] = 0
        predictionsTrain = [labelTrain(i)[0] for i in xrange(trainBatchNum)]
        predictions = [item for sublist in predictionsTrain for item in sublist]  # returns the flatten() list
        for j in xrange(len(predictions)):
            clusters[predictions[j]] += 1
        return clusters

    def getClusters(self, labelTrain, trainBatchNum, train_dev):
        clusters = {}
        for i in xrange(self.relationNum):
            clusters[i] = []
        predictionsTrain = [labelTrain(i)[0] for i in xrange(trainBatchNum)]
        predictions = [item for sublist in predictionsTrain for item in sublist]  # returns the flatten() list
        for j in xrange(len(predictions)):
            clusters[predictions[j]].append(self.data.getExampleRelation(j, train_dev))
        return clusters


    def getClusteredFreq(self, clusters):
        clustFreq = {}
        for i in xrange(self.relationNum):
            clustFreq[i] = {}
        j = 0
        for c in clusters:
            for feat in clusters[c]:
                if feat in clustFreq[j]:
                    clustFreq[j][feat] += 1
                else:
                    clustFreq[j][feat] = 1
            clustFreq[j] = sorted(clustFreq[j].iteritems(), key=operator.itemgetter(1), reverse=True)
            j += 1
        return clustFreq

    def printFirstK(self, k, clusterFreq):
        for c in clusterFreq:
            print clusterFreq[c][:k]


    def getClustersWithFrequencies(self, clusterSets, data, threshold):
        for c in clusterSets:
            frequency = {}
            print c,
            for elem in clusterSets[c]:
                trig = self.goldStandard['train'][elem][0]
                if trig in frequency:
                    frequency[trig] += 1
                else:
                    frequency[trig] = 1
            sorted_freq = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
            if len(sorted_freq) < threshold:
                for el in sorted_freq:
                    print el,
            else:
                count = 0
                for el in sorted_freq:
                    if count > threshold:
                        break
                    else:
                        print el,
                        count += 1
            print ''


def saveModel(model, name):
    pklProtocol = 2
    pklFile = open(settings.models_path + name, 'wb')
    pickle.dump(model, pklFile, protocol=pklProtocol)

def loadModel(name):
    pklFile = open(settings.models_path + name, 'rb')
    return pickle.load(pklFile)

def loadData(args, rng, negativeSamples, relationNum, modelType):
    """
    rng: random number generator
    """
    if not os.path.exists(args.pickled_dataset):
        print "Pickled dataset not found"
        sys.exit()

    tStart = time.time()
    print "Found existing pickled dataset, loading...",

    pklFile = open(args.pickled_dataset, 'rb')

    featureExtrs = pickle.load(pklFile)

    relationLexicon = pickle.load(pklFile)

    data = pickle.load(pklFile)

    goldStandard = pickle.load(pklFile)

    pklFile.close()
    tEnd = time.time()
    print "Done (" + str(tEnd - tStart) + "s.)"

    trigs = False


    indexedDataset = DataSetManager(data, relationLexicon, rng, negativeSamples, relationNum, trigs)

    print "Produced indexed dataset"

    return indexedDataset, goldStandard

def pickleClustering(clustering, clusteringName):
    pklProtocol = 2
    pklFile = open(settings.clusters_path + clusteringName, 'wb')
    pickle.dump(clustering, pklFile, protocol=pklProtocol)


def picklePosteriors(posteriors, posteriorsName):
    pklProtocol = 2
    pklFile = open(settings.clusters_path + posteriorsName, 'wb')
    pickle.dump(posteriors, pklFile, protocol=pklProtocol)

def getClustersWithInfo(clusterSets, data, threshold):
    for c in clusterSets:
        print c,
        if len(clusterSets[c]) < threshold:
            for elem in clusterSets[c]:
                print elem, data.getExampleFeatures(elem),
        else:
            count = 0
            for elem in clusterSets[c]:
                if count > threshold:
                    break
                else:
                    print elem, data.getExampleFeatures(elem),
                    count += 1
        print ''


def getClustersWithFrequencies(clusterSets, data, threshold):
    for c in clusterSets:
        frequency = {}
        print c,
        for elem in clusterSets[c]:
            trig = data.getExampleFeature(elem, 'trigger')
            if trig is not None:
                trig = trig.replace('trigger#', '')
                if trig in frequency:
                    frequency[trig] += 1
                else:
                    frequency[trig] = 1
        sorted_freq = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
        if len(sorted_freq) < threshold:
            for el in sorted_freq:
                print el,
        else:
            count = 0
            for el in sorted_freq:
                if count > threshold:
                    break
                else:
                    print el,
                    count += 1
        print ''


def getClustersWithFrequenciesValid(clusterSets, data, threshold):
    for c in clusterSets:
        frequency = {}
        print c,
        for elem in clusterSets[c]:
            trig = data.getExampleFeatureValid(elem, 'trigger')
            if trig is not None:
                trig = trig.replace('trigger#', '')
                if trig in frequency:
                    frequency[trig] += 1
                else:
                    frequency[trig] = 1
        sorted_freq = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
        if len(sorted_freq) < threshold:
            for el in sorted_freq:
                print el,
        else:
            count = 0
            for el in sorted_freq:
                if count > threshold:
                    break
                else:
                    print el,
                    count += 1
        print ''


def getClustersWithFrequenciesTest(clusterSets, data, threshold):
    for c in clusterSets:
        frequency = {}
        print c,
        for elem in clusterSets[c]:
            trig = data.getExampleFeatureTest(elem, 'trigger')
            if trig is not None:
                trig = trig.replace('trigger#', '')
                if trig in frequency:
                    frequency[trig] += 1
                else:
                    frequency[trig] = 1
        sorted_freq = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
        if len(sorted_freq) < threshold:
            for el in sorted_freq:
                print el,
        else:
            count = 0
            for el in sorted_freq:
                if count > threshold:
                    break
                else:
                    print el,
                    count += 1
        print ''

def getClustersWithRelationLabels(clusterSets, data, evaluation, threshold):
    for c in clusterSets:
        print c,
        if len(clusterSets[c]) < threshold:
            for elem in clusterSets[c]:
                if evaluation.relations[elem][0] != '':
                    print elem, data.getExampleFeatures(elem), evaluation.relations[elem],
        else:
            count = 0
            for elem in clusterSets[c]:
                if count > threshold:
                    break
                else:
                    if evaluation.relations[elem][0] != '':
                        print elem, data.getExampleFeatures(elem), evaluation.relations[elem],
                        count += 1
        print ''


def getCommandArgs():
    parser = argparse.ArgumentParser(description='Trains a basic Open Information Extraction Model')

    parser.add_argument('--pickled_dataset', metavar='pickled_dataset', nargs='?', required=True,
                        help='the pickled dataset file (produced by OiePreprocessor.py)')

    parser.add_argument('--epochs', metavar='epochs', nargs='?', type=int, default=100,
                        help='maximum number of epochs')

    parser.add_argument('--learning_rate', metavar='learning_rate', nargs='?', type=float, default=0.1,
                        help='initial learning rate')

    parser.add_argument('--batch_size', metavar='batch_size', nargs='?', type=int, default=50,
                        help='size of the minibatches')

    parser.add_argument('--embed_size', metavar='embed_size', nargs='?', type=int, default=30,
                        help='initial learning rate')

    parser.add_argument('--relations_number', metavar='relations_number', type=int, nargs='?', default=3,
                        help='number of relations to induce')

    parser.add_argument('--negative_samples_number', metavar='negative_samples_number', nargs='?', type=int, default=5,
                        help='number of negative samples')

    parser.add_argument('--l1_regularization', metavar='l1_regularization', nargs='?', type=float, default=0.0,
                        help='lambda value of L1 regulatization')

    parser.add_argument('--l2_regularization', metavar='l2_regularization', nargs='?', type=float, default=0.0,
                        help='lambda value of L2 regulatization')

    parser.add_argument('--optimization', metavar='optimization', nargs='?', type=int, default='0',
                        help='optimization algorithm 0 SGD, 1 ADAGrad, 2 ADADelta. Default SDG.')

    parser.add_argument('--model_name', metavar='model_name', nargs='?', required=True, type=str,
                        help='Name or ID of the model')

    parser.add_argument('--model', metavar='model', nargs='?', type=str, required=True,
                        help='Model Type choose among A, C, AC.')

    parser.add_argument('--fixed_sampling', metavar='fixed_sampling', nargs='?', default='False',
                        help='fixed/dynamic sampling switch, default fixed sampling')

    parser.add_argument('--ext_emb', metavar='ext_emb', nargs='?', default='False',
                        help='external embeddings, default False')

    parser.add_argument('--extended_reg', metavar='extended_reg', nargs='?', default='False',
                        help='extended regularization on reconstruction parameters, default False')

    parser.add_argument('--frequent_eval', metavar='frequent_eval', nargs='?', default='False',
                        help='using frequent evaluation, default False')

    parser.add_argument('--seed', metavar='seed', nargs='?', type=int, default=2,
                        help='random seed, default 2')

    parser.add_argument('--alpha', metavar='alpha', nargs='?', type=float, default=1.0,
                        help='alpha coefficient for scaling the entropy term')


    return parser.parse_args()





if __name__ == '__main__':
    print "Relation Learner"

    args = getCommandArgs()
    print args
    rseed = args.seed
    rand = np.random.RandomState(seed=rseed)


    negativeSamples = args.negative_samples_number
    numberRelations = args.relations_number
    indexedData, goldStandard = loadData(args, rand, negativeSamples, numberRelations, args.model)


    maxEpochs = args.epochs
    learningRate = args.learning_rate
    batchSize = args.batch_size
    embedSize = args.embed_size
    lambdaL1 = args.l1_regularization
    lambdaL2 = args.l2_regularization
    optimization = args.optimization
    modelName = args.model_name
    model = args.model
    fixedSampling = eval(args.fixed_sampling)
    extEmb = eval(args.ext_emb)
    extendedReg = eval(args.extended_reg)
    frequentEval = eval(args.frequent_eval)
    alpha = args.alpha
    inducer = ReconstructInducer(indexedData, goldStandard, rand, maxEpochs, learningRate,
                                 batchSize, embedSize, lambdaL1, lambdaL2, optimization, modelName,
                                 model, fixedSampling, extEmb, extendedReg,
                                 frequentEval, alpha)



    inducer.learn()

    saveModel(inducer, inducer.modelName)

