__author__ = 'diego'


import theano.tensor as T
import theano
from models.encoders.RelationClassifier import IndependentRelationClassifiers

class OieModelFunctions(object):

    def __init__(self, rng, featureDim, embedSize, relationNum, argVocSize, model,
                 rank, data, extEmb, extendedReg, alpha, parint):
        self.rng = rng

        self.h = featureDim
        self.k = embedSize
        self.r = relationNum
        self.n = rank
        self.a = argVocSize
        self.model = model
        self.relationClassifiers = IndependentRelationClassifiers(rng, featureDim, relationNum)
        self.params = self.relationClassifiers.params
        self.alpha = alpha
        self.parint = parint
        print 'Feature space size =', self.h
        print 'Argument vocabulary size =', argVocSize

        self.L1 = T.sum(abs(self.relationClassifiers.W))

        self.L2 = T.sum(T.sqr(self.relationClassifiers.W))  # + T.sum(T.sqr(self.relationClassifiers.Wb))

        if self.model == 'A':
            print 'Bilinear Model'
            from models.decoders.Bilinear import Bilinear

            self.argProjector = Bilinear(rng, embedSize, relationNum, self.a, data, extEmb, parint)
            self.params += self.argProjector.params
            if extendedReg:
                self.L1 += T.sum(abs(self.argProjector.C))
                self.L2 += T.sum(T.sqr(self.argProjector.C))

        elif self.model == 'AC':
            print 'Bilinear + Selectional Preferences Model'
            from models.decoders.BilinearPlusSP import BilinearPlusSP

            self.argProjector = BilinearPlusSP(rng, embedSize, relationNum, self.a, data, extEmb, parint)
            self.params += self.argProjector.params
            if extendedReg:
                self.L1 += T.sum(abs(self.argProjector.C1)) + T.sum(abs(self.argProjector.C2)) + T.sum(abs(self.argProjector.C))
                self.L2 += T.sum(T.sqr(self.argProjector.C1)) + T.sum(T.sqr(self.argProjector.C2)) + T.sum(T.sqr(self.argProjector.C))


        elif self.model == 'C':
            print 'Selectional Preferences'
            from models.decoders.SelectionalPreferences import SelectionalPreferences

            self.argProjector = SelectionalPreferences(rng, embedSize, relationNum, self.a, data, extEmb, parint)
            self.params += self.argProjector.params
            if extendedReg:
                self.L1 += T.sum(abs(self.argProjector.C1)) + T.sum(abs(self.argProjector.C2))
                self.L2 += T.sum(T.sqr(self.argProjector.C1)) + T.sum(T.sqr(self.argProjector.C2))



    def buildTrainErrComputation(self, batchSize, negNum, xFeats, args1, args2, trigs, neg1, neg2, negTrig):
        l = batchSize
        n = negNum

        # print xFeats
        print "Relation classifiers..."
        # relationLabeler.output are probabilities of relations assignment arranged in a tensor [l, r]
        relationProbs = self.relationClassifiers.compRelationProbsFunc(xFeats=xFeats)
        print "Arg projection..."

        entropy = self.alpha * -T.sum(T.log(relationProbs) * relationProbs, axis=1)  # [l,r] * [l,r] = [l]

        if self.model == 'A':
            allScores = self.argProjector.getScores(args1, args2, l, n, relationProbs, neg1, neg2, entropy)


        elif self.model == 'AC':
            allScores = self.argProjector.getScores(args1, args2, l, n, relationProbs, neg1, neg2, entropy)


        elif self.model == 'C':
            allScores = self.argProjector.getScores(args1, args2, l, n, relationProbs, neg1, neg2, entropy)


        resError = -T.mean(allScores)
        print "Done with building the graph..."
        # resError = theano.printing.Print("resError ")(resError)
        return resError




    def buildLabelComputation(self, batchSize, xFeats):
        #  xFeats [ l * e, h ] matrix
        return self.relationClassifiers.labelFunct(batchSize, xFeats)


    def buildRelationProbComputation(self, batchSize, xFeats):
        return self.relationClassifiers.compRelationProbsFunc(xFeats)

