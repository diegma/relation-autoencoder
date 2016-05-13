__author__ = 'enfry'

import math
import theano
from definitions import settings
import numpy as np
import theano.tensor as T
import cPickle as pickle

class SelectionalPreferences(object):

    def __init__(self, rng, embedSize, relationNum, argVocSize, data, ex_emb, parint=''):

        self.k = embedSize
        self.r = relationNum
        self.a = argVocSize

        a = self.a
        k = self.k
        r = self.r




        # Selectional Preferences
        Ca1NP = np.asarray(rng.normal(0, math.sqrt(0.1), size=(k, r)), dtype=theano.config.floatX)
        Ca2NP = np.asarray(rng.normal(0, math.sqrt(0.1), size=(k, r)), dtype=theano.config.floatX)
        self.C1 = theano.shared(value=Ca1NP, name='C1')
        self.C2 = theano.shared(value=Ca2NP, name='C2')

        # argument embeddings
        ANP = np.asarray(rng.uniform(-0.01, 0.01, size=(a, k)), dtype=theano.config.floatX)  # @UndefinedVariable

        if ex_emb:
            import gensim
            external_embeddings = gensim.models.Word2Vec.load(settings.external_embeddings_path)
            # for idArg in xrange(self.a):
            #     arg = data.id2Arg[idArg].lower().replace(' ', '_')
            #     if arg in external_embeddings:
            #         ANP[idArg] = external_embeddings[arg]
            for idArg in xrange(self.a):
                arg = data.id2Arg[idArg].lower().split(' ')
                new = np.zeros(k, dtype=theano.config.floatX)
                size = 0
                for ar in arg:
                    if ar in external_embeddings:
                        new += external_embeddings[ar]
                        size += 1
                if size > 0:
                    ANP[idArg] = new/size

        self.A = theano.shared(value=ANP, name='A')  # (a1, k)
        #self.A = theano.printing.Print("A=")(self.A)
        # selectional preference bias, do we need it?
        # self.Cb = theano.shared(value=np.zeros(k,  dtype=theano.config.floatX),  # @UndefinedVariable
        #                         name='Cb', borrow=True)

        # self.C2 = theano.printing.Print("C2=")(self.C2)
        # argument bias(as arguments are 'predicted' by the model)
        self.Ab = theano.shared(value=np.zeros(a,  dtype=theano.config.floatX),  # @UndefinedVariable
                                name='Ab', borrow=True)

        self.params = [self.A, self.C1, self.C2, self.Ab]





    def leftMostFactorization(self, batchSize, args, wC1):
        l = batchSize
        k = self.k  # embed size
        r = self.r  # relation number
        argEmbeds = self.A[args.flatten()]
        # first = T.dot(relationProbs, self.C1.dimshuffle(1, 0)) #+ self.Cb  # [l,r] * [r,k] = [l, k]
        Afirst = T.batched_dot(wC1, argEmbeds)
        return Afirst

    def rightMostFactorization(self, batchSize, args, wC2):
        l = batchSize
        k = self.k  # embed size
        r = self.r  # relation number
        argEmbeds2 = self.A[args.flatten()]
        # second = T.dot(relationProbs, self.C2.dimshuffle(1, 0))  # [l,r] * [r,k] = [l, k]
        Asecond = T.batched_dot(wC2, argEmbeds2)
        return Asecond



    def negLeftMostFactorization(self, batchSize, negEmbed, wC1):
        # l = batchSize
        # k = self.k  # embed size
        # r = self.r  # relation number
        # first = T.dot(relationProbs, self.C1.dimshuffle(1, 0))  # [l,r] * [r,k] = [l, k]
        Afirst = T.batched_tensordot(wC1, negEmbed.dimshuffle(1, 2, 0), axes=[[1], [1]])  # [l,k] [l,k,n] = [l,n]
        return Afirst

    def negRightMostFactorization(self, batchSize, negEmbed, wC2):
        # l = batchSize
        # k = self.k  # embed size
        # r = self.r  # relation number
        # second = T.dot(relationProbs, self.C2.dimshuffle(1, 0))  # [l,r] * [r,k] = [l, k]
        Asecond = T.batched_tensordot(wC2, negEmbed.dimshuffle(1, 2, 0), axes=[[1], [1]])  # [l,k] [l,k,n] = [l,n]
        return Asecond



    def getScores(self, args1, args2, l, n, relationProbs, neg1, neg2, entropy):
        weightedC1= T.dot(relationProbs, self.C1.dimshuffle(1, 0))
        weightedC2= T.dot(relationProbs, self.C2.dimshuffle(1, 0))

        left1 = self.leftMostFactorization(batchSize=l, args=args1, wC1=weightedC1)
        right1 = self.rightMostFactorization(batchSize=l, args=args2, wC2=weightedC2)
        one = left1 + right1

        u = T.concatenate([one + self.Ab[args1], one + self.Ab[args2]])
        logScoresP = T.log(T.nnet.sigmoid(u))
        allScores = logScoresP
        allScores = T.concatenate([allScores, entropy, entropy])

        negembed1 = self.A[neg1.flatten()].reshape((n, l, self.k))
        negembed2 = self.A[neg2.flatten()].reshape((n, l, self.k))
        negative1 = self.negLeftMostFactorization(batchSize=l,
                                                  negEmbed=negembed1,
                                                  wC1=weightedC1)
        negative2 = self.negRightMostFactorization(batchSize=l,
                                                  negEmbed=negembed2,
                                                  wC2=weightedC2)

        negOne = negative1.dimshuffle(1, 0) + right1
        negTwo = negative2.dimshuffle(1, 0) + left1
        g = T.concatenate([negOne + self.Ab[neg1], negTwo + self.Ab[neg2]])
        logScores = T.log(T.nnet.sigmoid(-g))
        allScores = T.concatenate([allScores, logScores.flatten()])

        return allScores


