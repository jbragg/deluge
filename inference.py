"""inference.py"""

import numpy as np
from scipy.misc import logsumexp

def infer(votes, labels, params, joint=False):
    """Performs inference.

    Inputs:
    - votes: label -> votes (lst: bool)
    - params: dict of parameters (see learning.py)
    - joint: use co-occurrence probabilities (bool)

    Outputs: P(label | votes), log likelihood
    """
    
    #---- compute P(v | L) ------
    pV = params['pV'] # probability worker is right

    vGl = dict()

    for l in labels:
        ss_true = sum(votes[l])
        ss_false = len(votes[l]) - ss_true
        pos = np.log(pV[0]) * ss_true + np.log(1-pV[0]) * ss_false
        neg = np.log(pV[1]) * ss_false + np.log(1-pV[1]) * ss_true
        vGl[l] = (pos, neg)

    # convert to numpy array
    vGl = np.array([[vGl[l][0] for l in labels], [vGl[l][1] for l in labels]])

    #-- predict for individual labels --
    pL = params['pL'] # dict
    pL = np.log(np.vstack((pL,1-pL)))
    vAl = pL + vGl
    
    # P(l | v)
    # normalize in exp space
    ll = logsumexp(vAl,0)
    lGv = vAl - ll

    # return if not joint
    q = np.exp(lGv)
    if not joint:
        return q, np.sum(ll)

    #--- predict using co-occurrences ---
    vAljoint = dict()
    lCorrPos = params['lCorrPos']
    lCorrNeg = params['lCorrNeg']
    
    vAlJointPos0 = vGl[0,:] + np.log(lCorrPos)
    vAlJointPos1 = vGl[1,:] + np.log(1-lCorrPos)
    vAlJointNeg0 = vGl[0,:] + np.log(lCorrNeg)
    vAlJointNeg1 = vGl[1,:] + np.log(1-lCorrNeg)

    # fix diagonals (just for matrices with nonzero diagonal)
    d = len(labels)
    vAlJointPos0[xrange(d),xrange(d)] += pL[0,:]
    vAlJointNeg1[xrange(d),xrange(d)] += pL[1,:]

    vAlJoint = np.vstack((np.sum(np.logaddexp(vAlJointPos0,vAlJointPos1),1),
                          np.sum(np.logaddexp(vAlJointNeg0,vAlJointNeg1),1)))

    lGvJoint = vAlJoint - logsumexp(vAlJoint,0)

    return np.exp(lGvJoint), None
