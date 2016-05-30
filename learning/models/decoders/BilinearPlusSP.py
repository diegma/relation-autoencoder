__author__ = 'enfry'

import math
import theano
from definitions import settings
import numpy as np
import theano.tensor as T
import cPickle as pickle

class BilinearPlusSP(object):

    def __init__(self, rng, embedSize, relationNum, argVocSize, data, ex_emb, ):

        self.k = embedSize
        self.r = relationNum
        self.a = argVocSize

        a = self.a
        k = self.k
        r = self.r


        # KxK matrix for each argument-argument for each relation
        CNP = np.asarray(rng.normal(0, math.sqrt(0.1), size=(k, k, r)), dtype=theano.config.floatX)
                                                                                            # @UndefinedVariable
        self.C = theano.shared(value=CNP, name='C')
        # self.C = theano.printing.Print("C = ")(self.C)

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

        self.Ab = theano.shared(value=np.zeros(a,  dtype=theano.config.floatX),  # @UndefinedVariable
                                 name='Ab', borrow=True)

        self.params = [self.C, self.A, self.Ab, self.C1, self.C2]




    def factorization(self, batchSize, argsEmbA, argsEmbB, wC, wC1, wC2):
        # l = batchSize
        # k = self.k  # embed size
        # r = self.r  # relation number

        Afirst = T.batched_tensordot(wC, argsEmbA, axes=[[1], [1]])  # + self.Cb  # [l, k, k] * [l, k] = [l, k]
        Asecond = T.batched_dot(Afirst, argsEmbB)  # [l, k] * [l, k] = [l]
        spFirst = T.batched_dot(wC1, argsEmbA)
        spSecond = T.batched_dot(wC2, argsEmbB)
        return Asecond + spFirst + spSecond



    def negLeftFactorization(self, batchSize, negEmbA, argsEmbB, wC, wC1, wC2):
        # l = batchSize
        # k = self.k  # embed size
        # r = self.r  # relation number

        Afirst = T.batched_tensordot(wC, negEmbA.dimshuffle(1, 2, 0), axes=[[1], [1]])  # [l, k, k] * [n, l, k] = [l, k, n]
        Asecond = T.batched_tensordot(Afirst, argsEmbB, axes=[[1], [1]])  # [l, k, n] * [l, k] = [l, n]

        spAfirst = T.batched_tensordot(wC1, negEmbA.dimshuffle(1, 2, 0), axes=[[1], [1]])  # [l,k] [l,k,n] = [l,n]

        spSecond = T.batched_dot(wC2, argsEmbB)

        return Asecond + spAfirst + spSecond.reshape((batchSize, 1))

    def negRightFactorization(self, batchSize, argsEmbA, negEmbB, wC, wC1, wC2):
        Afirst = T.batched_tensordot(wC, argsEmbA, axes=[[1], [1]])  # [l, k, k] * [l, k] = [l, k]
        Asecond = T.batched_tensordot(Afirst, negEmbB.dimshuffle(1, 2, 0), axes=[[1], [1]])  # [l, k] * [l, k, n] = [l, n]
        spFirst = T.batched_dot(wC1, argsEmbA)
        spAsecond = T.batched_tensordot(wC2, negEmbB.dimshuffle(1, 2, 0), axes=[[1], [1]])  # [l,k] [l,k,n] = [l,n]
        return Asecond + spAsecond + spFirst.reshape((batchSize, 1))



    def getScores(self, args1, args2, l, n, relationProbs, neg1, neg2, entropy):
        weightedC1 = T.dot(relationProbs, self.C1.dimshuffle(1, 0))
        weightedC2 = T.dot(relationProbs, self.C2.dimshuffle(1, 0))
        weightedC = T.tensordot(relationProbs, self.C, axes=[[1], [2]])


        argembed1 = self.A[args1]
        argembed2 = self.A[args2]

        one = self.factorization(batchSize=l,
                                 argsEmbA=argembed1,
                                 argsEmbB=argembed2,
                                 wC=weightedC,
                                 wC1=weightedC1,
                                 wC2=weightedC2)

        u = T.concatenate([one + self.Ab[args1], one + self.Ab[args2]])
        logScoresP = T.log(T.nnet.sigmoid(u))

        allScores = logScoresP
        allScores = T.concatenate([allScores, entropy, entropy])


        negembed1 = self.A[neg1.flatten()].reshape((n, l, self.k))
        negembed2 = self.A[neg2.flatten()].reshape((n, l, self.k))
        negOne = self.negLeftFactorization(batchSize=l,
                                           negEmbA=negembed1,
                                           argsEmbB=argembed2,
                                           wC=weightedC,
                                           wC1=weightedC1,
                                           wC2=weightedC2)

        negTwo = self.negRightFactorization(batchSize=l,
                                            argsEmbA=argembed1,
                                            negEmbB=negembed2,
                                            wC=weightedC,
                                            wC1=weightedC1,
                                            wC2=weightedC2)
        g = T.concatenate([negOne + self.Ab[neg1].dimshuffle(1, 0),
                           negTwo + self.Ab[neg2].dimshuffle(1, 0)])
        logScores = T.log(T.nnet.sigmoid(-g))
        allScores = T.concatenate([allScores, logScores.flatten()])

        return allScores

