__author__ = 'diego'


import numpy as np


class NegativeExampleGenerator(object):
    def __init__(self, rand, negSamplingCum):
        self._rand = rand
        self._negSamplingCum = negSamplingCum
        # self._neg2SamplingCum = neg2SamplingCum
#         self._negSamplingDistrPower = negSamplingDistrPower
#         self._compute_unigram_distribution()

    def _univariate_distr_sample(self, sampleSize=1):
        return [self._negSamplingCum.searchsorted(self._rand.uniform(0, self._negSamplingCum[-1]))
                for i in xrange(0, sampleSize)]

    def generate_random_negative_example(self, positiveArgs, negativeExampleNum):
        l = positiveArgs.shape[0]  # number of positive instances
        n = negativeExampleNum  # number of negative examples generated per instance

        negativeArgs = np.zeros((n, l), dtype=np.int32)
        for instance_idx in xrange(l):
            samples = self._univariate_distr_sample(n)
            for negNum_idx in xrange(n):
                negativeArgs[negNum_idx, instance_idx] = samples[negNum_idx]
        return negativeArgs
