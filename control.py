from __future__ import division
import collections
import random
#from preprocess import cats as labels
import inference
import util
from util import entropy
import numpy as np
import copy


SAMPLE_NUM = 100

def compute_belief(votes_acc, labels, params, d):
    def infer(votes_dict):

        if d['inf'] == 'maj':        
            votes = np.array([util.maj_vote(votes_acc[l]) for l in labels])
            return np.vstack((votes, 1-votes))
        elif d['inf'] == 'ind':
            return inference.infer(votes_dict, labels, params)[0]
        elif d['inf'] == 'joint':
            return inference.infer(votes_dict, labels, params, True)
        elif d['inf'] == 'bp':
            return inference.loopy(votes_dict, labels, params)[0]
        else:
            raise Exception

    belief = infer(votes_acc)
    return belief


def control(votes_acc, votes, labels, params, d, n):
    remaining = dict((l, len(votes[l])-len(votes_acc[l])) for l in labels)
    selected = []
    try:
        method = d['batch']
    except:
        method = 'k'

    # compute initial beliefs
    belief, heuristic = selectnext(votes_acc, votes, labels, [], params, d)
    belief_dict = util.np_to_belief(belief, labels)
    sortedLabels = sorted([l for l in labels if remaining[l]>0],
                          key=lambda x: heuristic[x])

    selected_labels = dict()
    assert len(sortedLabels) >= n
    if method == 'k':
        next_l = sortedLabels[:n]
        for l in next_l:
            selected.append((l, votes[l][len(votes[l])-remaining[l]]))
            remaining[l] -= 1
            selected_labels[l] = True
    elif method == 'MAP' or method == 'sample':
        # add first label
        next_l = sortedLabels[0]
        selected.append(
            (next_l, votes[next_l][len(votes[next_l])-remaining[next_l]]))
        remaining[next_l] -= 1
        selected_labels[next_l] = True

        # compute remaining labels:
        for i in xrange(n-1):
            _, heuristic = selectnext(votes_acc, votes, labels, selected,
                                      params, d, belief)
            next_l = min([l for l in labels if
                          remaining[l]>0 and l not in selected_labels],
                         key=lambda x: heuristic[x])
            selected.append(
                (next_l, votes[next_l][len(votes[next_l])-remaining[next_l]]))
            remaining[next_l] -= 1
            selected_labels[next_l] = True
    else:
        raise Exception
    
    # ensure we do not have repeated labels (can't ask someone twice)
    selected_labels = [l for l,v in selected]
    assert len(selected_labels) == len(set(selected_labels))

    return belief_dict, selected

def predict_votes(belief, wModel):
    return np.sum(belief * np.array([[wModel[0,0]],[wModel[0,1]]]),0)

def selectnext(votes_acc, votes, labels, selected, params, d, b=None):
    """ function returns belief and ordered list of next label to select 
    according to
    heuristic defined in method

    BUG: outdated description

    """
    
    
    if b is None:
        belief = compute_belief(votes_acc, labels, params, d)
    else:
        belief = b
    old_belief = None

    # wModel = [t|t, t|f
    #           f|t, f|f]
    tmp = np.array([params['pV'][0], 1-params['pV'][1]])
    wModel = np.vstack((tmp, 1-tmp))
 
    if 'batch' in d and d['batch'] == 'MAP':
        if selected:
            # BUG: double check this
            pV = predict_votes(belief, wModel)
            votes_acc_new = copy.deepcopy(votes_acc)
            for i,l in enumerate(labels):
                if l in [l for l,v in selected]:
                    votes_acc_new[l].append(pV[i] >= 0.5)
            old_belief = belief
            belief = compute_belief(votes_acc_new, labels, params, d)
    

    #------ random layers ------
    if d['order'] == 'layer':

        heuristic = dict((l,len(votes_acc[l])) for l in labels)

    #------ VOI ------
    elif d['order'] == 'voi':
        heuristic = util.expectimax(votes_acc, labels, params, belief,
                                    infer, max_depth=0)

    #------ entropy ------
    elif d['order'] == 'entropy':
        # negative entropy 
        heuristic = dict((l,-1*entropy(belief[:,i])) for 
                         i,l in enumerate(labels))

        # proximity to 0.5 belief (equivalent)
#        heuristic = dict((l,abs(0.5 - belief_dict[l])) for l in labels)

    #------ joint entropy maximize (greedy) ------
    elif d['order'] == 'entropy2':
        # H(X | A) = H(X)
        heuristic = dict((l,-1*entropy(np.sum(belief[:,i]*wModel,1))) for 
                         i,l in enumerate(labels))

    #------ voi maximize (greedy) ------
    elif d['order'] == 'voi_subm':
        heuristic = dict()

        # sample conditional entropy
        if selected and d['batch'] == 'sample':
            sample = True
            hx_dict = conditional_entropy(belief, wModel, labels,
                                          params, d,
                                          votes_acc, selected, SAMPLE_NUM)
        else:
            sample = False

        for i,l in enumerate(labels):
            # P(V, L)
            pVL = belief[:,i] * wModel

            if sample:
                # H(X | A)
                hx = hx_dict[l]
            else:
                # H(X | A) = H(X)
                hx = entropy(np.sum(pVL, 1))

            # H(X | U) 
            hxu = -1*np.sum(pVL * np.log(wModel))

            # H(X | A) - H(X | U)
            heuristic[l] = -1 * (hx - hxu) # change to argmin


    else:
        raise Exception('Undefined order')


    return belief, heuristic

def conditional_entropy(belief, wModel, labels,
                        params, d,
                        votes_acc, selected, N):
    pV = predict_votes(belief, wModel)

    H = collections.defaultdict(float)
    for it in xrange(N):
        vote_sample = np.random.rand(len(labels)) <= pV
        votes_acc_new = copy.deepcopy(votes_acc)
        for i,l in enumerate(labels):
            if l in [l for l,v in selected]:
                votes_acc_new[l].append(vote_sample[i])
        b = compute_belief(votes_acc_new, labels, params, d)
        pV_new = predict_votes(b, wModel)
        pV_new = np.vstack((pV_new, 1-pV_new))
        for i,l in enumerate(labels):
            H[l] += 1/N * entropy(pV_new[:,i])

    return H
