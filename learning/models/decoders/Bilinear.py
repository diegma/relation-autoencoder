__author__ = 'enfry'

import math
import theano
from definitions import settings
import numpy as np
import theano.tensor as T
from collections import OrderedDict
import cPickle as pickle

class Bilinear(object):

    def __init__(self, rng, embedSize, relationNum, argVocSize, data, ex_emb):

        self.k = embedSize
        self.r = relationNum
        self.a = argVocSize

        a = self.a
        k = self.k
        r = self.r



        # KxK matrix for each argument-argument for each relation
        CNP = np.asarray(rng.normal(0, math.sqrt(0.1), size=(k, k, r)), dtype=theano.config.floatX)


        self.C = theano.shared(value=CNP, name='C')
        # self.C = theano.printing.Print("C = ")(self.C)
                # argument embeddings
        ANP = np.asarray(rng.uniform(-0.01, 0.01, size=(a, k)), dtype=theano.config.floatX)

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

        self.updates = OrderedDict({self.A: self.A / T.sqrt(T.sum(T.sqr(self.A), axis=0))})
        self.normalize = theano.function([], [], updates=self.updates)

        # self.params = [self.C, self.A]
        self.params = [self.C, self.A, self.Ab]



    def factorization(self, batchSize, argsEmbA, argsEmbB, wC):

        # first = T.tensordot(relationProbs, self.C, axes=[[1], [2]])  # [l,r] * [k,k,r] = [l, k, k]
        Afirst = T.batched_tensordot(wC, argsEmbA, axes=[[1], [1]])  # [l, k, k] * [l, k] = [l, k]
        Asecond = T.batched_dot(Afirst, argsEmbB)  # [l, k] * [l, k] = [l]
        # entropy = T.sum(T.log(relationProbs) * relationProbs, axis=1)  # [l,r] * [l,r] = [l]
        return Asecond

    def negFactorization1(self, batchSize, negEmbA, argsEmbB, wC):
        # first = T.tensordot(relationProbs, self.C, axes=[[1], [2]])  # [l,r] * [k,k,r] = [l, k, k]
        Afirst = T.batched_tensordot(wC, negEmbA.dimshuffle(1, 2, 0), axes=[[1], [1]])  # [l, k, k] * [n, l, k] = [l, k, n]
        Asecond = T.batched_tensordot(Afirst, argsEmbB, axes=[[1], [1]])  # [l, k, n] * [l, k] = [l, n]
        return Asecond

    def negFactorization2(self, batchSize, argsEmbA, negEmbB, wC):
        # first = T.tensordot(relationProbs, self.C, axes=[[1], [2]])  # [l,r] * [k,k,r] = [l, k, k]
        Afirst = T.batched_tensordot(wC, argsEmbA, axes=[[1], [1]])  # [l, k, k] * [l, k] = [l, k]
        Asecond = T.batched_tensordot(Afirst, negEmbB.dimshuffle(1, 2, 0), axes=[[1], [1]])  # [l, k] * [l, k, n] = [l, n]
        return Asecond


    def getScores(self, args1, args2, l, n, relationProbs, neg1, neg2, entropy):
        argembed1 = self.A[args1]
        argembed2 = self.A[args2]

        weightedC = T.tensordot(relationProbs, self.C, axes=[[1], [2]])
        one = self.factorization(batchSize=l,
                                 argsEmbA=argembed1,
                                 argsEmbB=argembed2,
                                 wC=weightedC)  # [l,n]

        u = T.concatenate([one + self.Ab[args1], one + self.Ab[args2]])

        logScoresP = T.log(T.nnet.sigmoid(u))

        allScores = logScoresP
        allScores = T.concatenate([allScores, entropy, entropy])


        negembed1 = self.A[neg1.flatten()].reshape((n, l, self.k))
        negembed2 = self.A[neg2.flatten()].reshape((n, l, self.k))
        negOne = self.negFactorization1(batchSize=l,
                                        negEmbA=negembed1,
                                        argsEmbB=argembed2,
                                        wC=weightedC)

        negTwo = self.negFactorization2(batchSize=l,
                                        argsEmbA=argembed1,
                                        negEmbB=negembed2,
                                        wC=weightedC)

        g = T.concatenate([negOne + self.Ab[neg1].dimshuffle(1, 0),
                           negTwo + self.Ab[neg2].dimshuffle(1, 0)])
        logScores = T.log(T.nnet.sigmoid(-g))
        allScores = T.concatenate([allScores, logScores.flatten()])
        return allScores


