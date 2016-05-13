__author__ = 'diego'

import numpy as np
import theano
import theano.tensor as T



class AdaGrad(object):
    def __init__(self, params):
        self.accumulator = []
        for para_i in params:
            eps_p = np.zeros_like(para_i.get_value(borrow=True), dtype=theano.config.floatX)
            self.accumulator.append(theano.shared(eps_p, borrow=True))

    def update(self, learningRate, params, cost):
        print 'AdaGrad takes the floor'
        grads = T.grad(cost, params)
        updates = []
        for param_i, grad_i, acc_i in zip(params, grads, self.accumulator):
            acc = acc_i + T.sqr(grad_i)
            updates.append((param_i, param_i - learningRate * grad_i / (T.sqrt(acc)+1e-6)))
            updates.append((acc_i, acc))
        return updates


class SGD(object):
    def update(self, learningRate, params, cost):
        print 'SGD takes the floor'
        grads = T.grad(cost, params)
        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learningRate * grad_i))
        return updates



