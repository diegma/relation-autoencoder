__author__ = 'diego'

class OieExample (object):
    """
    A container for a sentence: entities,k list of feature ids, trigger, (optional) relation label
    """
    def __init__(self, arg1, arg2, features, trigger, relation=''):
        self.features = features
        self.arg1 = arg1  # str
        self.arg2 = arg2  # str
        self.relation = relation
        self.trigger = trigger

    def setFeatures(self, features):
        self.features = features