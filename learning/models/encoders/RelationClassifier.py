__author__ = 'diego'

from theano import sparse
import theano
from definitions import settings
import numpy as np
import theano.tensor as T

class IndependentRelationClassifiers(object):
    # rng is a random generator,
    # featureDim is the dimension of the feature space
    # relationNum is the number of possible relations (classes of relations)

    def __init__(self, rng, featureDim, relationNum):

        # dimensionality of feature space
        self.h = featureDim
        # relation num
        self.r = relationNum
        # print str(np.sqrt(6. / (self.h + self.r)))
        # w_bound = np.sqrt(self.h * self.r)

        # print str(1.0 / w_bound)
        print 'low bound =', settings.low, 'high bound =', settings.high
        self.W = theano.shared(np.asarray(rng.uniform(
            low=settings.low,
            high=settings.high,
            size=(self.h, self.r)), dtype=theano.config.floatX),  # @UndefinedVariable
            name='W', borrow=True)
        # npW = np.zeros((3,3),dtype=theano.config.floatX)
        # npW[0,0] = 1.e+40
        # npW[1,1] = 1.e+40
        # npW[2,2] = 1.e+40

                                                                                            # @UndefinedVariable
        # self.W = theano.shared(value=np.asarray(npW))

        self.Wb = theano.shared(value=np.zeros(self.r,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                               name='Wb', borrow=True)

        self.params = [self.W, self.Wb]
        # self.params = [self.Wb]
        # self.params = []

    def compRelationProbsFunc(self, xFeats):
        #  xFeats [l, h] matrix
        # xFeats = theano.printing.Print("xFeats")(xFeats)
        # self.Wb = theano.printing.Print("Wb ") (self.Wb)
        # self.W = theano.printing.Print("W ") (self.W)
        # scores of each role by a classifier
        relationScores = sparse.dot(xFeats, self.W) + self.Wb   # [l, h] x [h, r] => [l, r]
        #relationScores = theano.printing.Print("relationScores=")(relationScores)

        # convert it to probabilities
        relationProbs = T.nnet.softmax(relationScores)
        #relationProbs = theano.printing.Print("relationProbs = ")(relationProbs)


        return relationProbs  # [l, r]


    def labelFunct(self, batchSize, xFeats):
        #  xFeats [l, h]
        # l = batchSize
        # self.W = theano.printing.Print("W ") (self.W)
        # self.Wb = theano.printing.Print("Wb ") (self.Wb)
        scores = sparse.dot(xFeats, self.W) + self.Wb  # [l, h] x [h, r] => [l, r]
        relationProbs = T.nnet.softmax(scores)
        # scores = theano.printing.Print("scores ") (scores)
        labels = T.argmax(scores, axis=1)  #  [l, r] => [l]
        # labels = theano.printing.Print("labels ") (labels)
        return (labels, relationProbs)