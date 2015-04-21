import collections
import numpy as np
import random
import itertools
import operator
import copy

class prettyfloat(float):
    def __repr__(self):
        return '%0.5f' % self

def np_to_belief(np_array,labels):
    """ takes first row of two-row belief np array and converts it to dict
        indexed by label of positive beliefs """
    return dict((l,np_array[0,i]) for i,l in enumerate(labels))

def acc_vote(truth, acc_true, acc_false):
    v = random.random()
    if truth and v <= acc_true:
        return truth
    elif truth:
        return not truth
    elif not truth and v <= acc_false:
        return truth
    else:
        return not truth


def thresh_vote(lst, f):
    """ takes a list of votes and predicts based on threshold
        returns true iff fraction of true votes >= f
    """

    if len(lst) == 0: # guess 0 by default (appropriate for our dataset)
        q = 0
    else:
        q = float(sum(lst)) / len(lst)

    return q >= f

def maj_vote(lst):
    """ performs majority vote """
    return thresh_vote(lst, 0.5)

#def accuracy(iter1,iter2):
#    """ returns average accuracy of entries from two iterables """
#    n = 0
#    matches = 0
#    for i,j in zip(iter1,iter2):
#        n += 1
#        matches += i == j
#
#    return matches / n
        
def dict_vals_by_sorted_key(d):
    return [d[k] for k in sorted(d.keys())]

def score_pred(pred,gt):
    """ returns average f-score """
    assert len(pred) == len(gt)

    if type(pred) is dict:
        assert type(gt) is dict
        pred = dict_vals_by_sorted_key(pred) 
        gt = dict_vals_by_sorted_key(gt)
    
#    for i in range(len(pred)):
#        print pred[i],gt[i]

    tp = sum(1 for x,y in zip(pred,gt) if x and y) 
    tn = sum(1 for x,y in zip(pred,gt) if not x and not y) 
    fp = sum(1 for x,y in zip(pred,gt) if x and not y) 
    fn = sum(1 for x,y in zip(pred,gt) if not x and y) 

    if tp+fp == 0:
        precision = 1
    else:
        precision = float(tp) / (tp+fp)

    if tp+fn == 0:
        recall = 1
    else:
        recall = float(tp) / (tp+fn)

#    print 'precision: %f recall: %f' % (precision,recall)

#    print
#    print 'tp: %d' % tp
#    print 'tn: %d' % tn
#    print 'fp: %d' % fp
#    print 'fn: %d' % fn

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

#    accuracy = float(len([x for x,y in zip(pred,gt) if x == y])) / len(pred)
    accuracy = float(tp+tn) / (tp+tn+fp+fn)

    return accuracy,precision,recall,f_score
 
#def score_acc(pred,gt):
#    """ returns average f-score """
#    assert len(pred) == len(gt)
#
#    if type(pred) is dict:
#        assert type(gt) is dict
#        pred = dict_vals_by_sorted_key(pred) 
#        gt = dict_vals_by_sorted_key(gt)
#
#    accuracy = float(len([x for x,y in zip(pred,gt) if x == y])) / len(pred)
#
#    return accuracy
        

#----------------- from before ------------
### general util functions
def approx_equal(x, y, tolerance=0.001):
    return abs(x-y) <= 0.5 * tolerance * (x + y)

def set_approx(l):
    acc = []
    for i in l:
        in_set = False
        for a in acc:
            if approx_equal(i,a,.000000001):
                in_set = True

        if not in_set:
            acc.append(i)

    return acc

### cascade-specific functions
def confusion_matrix(predicted, gt):
    """
    Takes dictionaries of predicted and ground truth and
    returns confusion matrix
    """
    tp = [k for k in predicted if predicted[k] and gt[k]]
    tn = [k for k in predicted if not predicted[k] and not gt[k]]
    fp = [k for k in predicted if predicted[k] and not gt[k]]
    fn = [k for k in predicted if not predicted [k] and gt[k]]

    return tp, tn, fp, fn

def item_answers(d):
    items = set(i for i,l in d if d[i,l])
    return dict((item, [l for i,l in d if d[i,l] and i==item]) \
                                                            for item in items)

def label_answers(d):
    labels = set(l for i,l in d if d[i,l])
    return dict((label, [i for i,l in d if d[i,l] and l==label]) \
                                                            for label in labels)

def true_labels(d):
    return [k for k in d if d[k]]

def dict_combinations(d):
    return [dict(p for p in zip(d.keys(),l)) for l in itertools.product(*d.values())]

#---------------------- NEW ------------------

def entropy(a):
    return np.sum(-1 * a * np.log(a))

def argmax(d): 
    """Returns argmax, max of dictionary"""
    return max(d.iteritems(), key=operator.itemgetter(1))

def bootstrap(data,func,nboot):
    """Produce nboot bootstrap samples from applying func to data

    Taken from http://www.cs.colostate.edu/~anderson/cs545/index.html/doku.php?id=notes:noteslinearmodelsbootstrap

    """

    n = len(data)
    resamples = np.array([[random.choice(data) for i in range(n)]
                          for j in range(nboot)])
    return np.apply_along_axis(func, 1, resamples)

def expectimax(votes, labels, params, belief, inf_f, depth=0, max_depth=1):
    pV = params['pV']
    pVotes = np.vstack((belief[0,:] * pV[0] + belief[1,:] * (1-pV[1]),
                        belief[1,:] * pV[1] + belief[0,:] * (1-pV[0])))

    exp_entropy = dict()

    # calculate expected entropy after asking about a label
    # BUG: assumes label independence
    for i,l in enumerate(labels):
        v = 0
    
        siz = sum(len(x) for x in votes.values())
        votes_exp = copy.deepcopy(votes)
        votes_exp[l].append(True)
        b1 = inf_f(votes_exp)
        if depth == max_depth:
            v += pVotes[0,i] * entropy(b1)
        else:
            v += pVotes[0,i] * argmax(expectimax(votes_exp, labels,
                                                 params, b1, inf_f,
                                                 depth=depth+1,
                                                 max_depth=max_depth))[1]

        votes_exp = copy.deepcopy(votes)
        votes_exp[l].append(False)
        b2 = inf_f(votes_exp)
        if depth == max_depth:
            v += pVotes[1,i] * entropy(b2)
        else:
            v += pVotes[1,i] * argmax(expectimax(votes_exp, labels,
                                                 params, b2, inf_f,
                                                 depth=depth+1,
                                                 max_depth=max_depth))[1]

        # ensure that original dict didn't change
        siz2 = sum(len(x) for x in votes.values())
        assert siz2 == siz

        exp_entropy[l] = v


    return exp_entropy



