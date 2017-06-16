__author__ = 'diego'

import nltk
import re, string
import settings
import pickle

parsing = 0
entities = 1
trig = 2
sentence = 3
pos = 4
docPath = 5
#  ======= Relation features =======
stopwords_list = nltk.corpus.stopwords.words('english')
_digits = re.compile('\d')
def bow(info, arg1, arg2):
    return info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()

def bow_clean(info, arg1, arg2):
    bow = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    result = []
    tmp = []
    for word in bow:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    for word in tmp:
        if word not in stopwords_list and not _digits.search(word) and not word[0].isupper():
            result.append(word)
    return result

def before_arg1(info, arg1, arg2):
    before = info[sentence][:info[sentence].find(arg1)]
    beforeSplit = before.lower().strip().split(' ')
    beforeSplit = [word for word in beforeSplit if word not in string.punctuation]
    # print beforeSplit
    if len(beforeSplit) > 1:
        return [beforeSplit[-2], beforeSplit[-1]]
    elif len(beforeSplit) == 1:
        if beforeSplit[0] != '':
            return [beforeSplit[-1]]
        else:
            return []
    else:
        return []


def after_arg2(info, arg1, arg2):
    after = info[sentence][info[sentence].rfind(arg2)+len(arg2):]
    afterSplit = after.lower().strip().split(' ')
    afterSplit = [word for word in afterSplit if word not in string.punctuation]
    if len(afterSplit) > 1:
        return [a for a in afterSplit[0: 2]]
    elif len(afterSplit) == 1:
        if afterSplit[0] != '':
            return [afterSplit[0]]
        else:
            return []
    else:
        return []

def bigrams(info, arg1, arg2):
    between = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    tmp = []
    for word in between:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    return [x[0]+'_'+x[1] for x in zip(tmp, tmp[1:])]

def trigrams(info, arg1, arg2):
    between = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    tmp = []
    for word in between:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    return [x[0]+'_'+x[1]+'_'+x[2] for x in zip(tmp, tmp[1:], tmp[2:])]

def skiptrigrams(info, arg1, arg2):
    between = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    tmp = []
    for word in between:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    return [x[0]+'_X_'+x[2] for x in zip(tmp, tmp[1:], tmp[2:])]

def skipfourgrams(info, arg1, arg2):
    between = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    tmp = []
    for word in between:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    return [x[0]+'_X_'+x[2] + '_' + x[3] for x in zip(tmp, tmp[1:], tmp[2:], tmp[3:])] +\
           [x[0]+'_'+x[1]+'_X_' + x[3] for x in zip(tmp, tmp[1:], tmp[2:], tmp[3:])]

def trigger(info, arg1, arg2):
    return info[trig].replace('TRIGGER:', '')

def entityTypes(info, arg1, arg2):
    return info[entities]

def entity1Type(info, arg1, arg2):
    return info[entities].split('-')[0]

def entity2Type(info, arg1, arg2):
    return info[entities].split('-')[1]

def arg1(info, arg1, arg2):
    return arg1

def arg1_lower(info, arg1, arg2):
    return arg1.lower()

def arg1unigrams(info, arg1, arg2):
    return arg1.lower().split()

def arg2(info, arg1, arg2):
    return arg2

def arg2_lower(info, arg1, arg2):
    return arg2.lower()

def arg2unigrams(info, arg1, arg2):
    return arg2.lower().split()

def lexicalPattern(info, arg1, arg2):
    # return info[parsing]
    p = info[parsing].replace('->', ' ').replace('<-', ' ').split()
    result = []
    for num, x in enumerate(p):
        if num % 2 != 0:
            result.append(x)
    return '_'.join(result)

def dependencyParsing(info, arg1, arg2):
    return info[parsing]


def rightDep(info, arg1, arg2):
    p = info[parsing].replace('->', ' -> ').replace('<-', ' <- ').split()
    return ''.join(p[:3])

def leftDep(info, arg1, arg2):
    p = info[parsing].replace('->', ' -> ').replace('<-', ' <- ').split()
    return ''.join(p[-3:])

def posPatternPath(info, arg1, arg2):
    words = info[sentence].split()
    postags = info[pos].split()
    assert len(postags) == len(words), 'error'
    a = []
    for w in xrange(len(words)):
        a.append((words[w], postags[w]))
    # a = info[4].split()
    if a:
        # print arg1, words
        # print [a.index(item) for item in a if item[0] == arg1.split()[-1]],'aaaaaaa'
        beginList = [a.index(item) for item in a if item[0] == arg1.split()[-1]]
        # print beginList
        endList = [a.index(item) for item in a if item[0] == arg2.split()[0]]
        # print endList
        if len(beginList) > 0 and len(endList) > 0:
            # posPattern = [item[1] for item in a if beginList[0] > a.index(item) > endList[0]]
            posPattern = []
            for num, item in enumerate(a):
                if beginList[0] < num < endList[0]:
                    posPattern.append(item[1])
            # print posPattern
            return '_'.join(posPattern)
        else:
            return ''
    else:
        return ''


def getBasicCleanFeatures():
    """
    "Rich features"
    :return: functions of [trigger, entityTypes, arg1_lower, arg2_lower, bow_clean, entity1Type, entity2Type, lexicalPattern,
                posPatternPath]
    """
    features = [trigger, entityTypes, arg1_lower, arg2_lower, bow_clean, entity1Type, entity2Type, lexicalPattern,
                posPatternPath]
    return features

