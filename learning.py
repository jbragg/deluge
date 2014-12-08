"""learning.py"""

from __future__ import division
import collections
import numpy as np
import scipy.stats
import inference

MAX_EM_ITERS = 200
MIN_EM_CHANGE = 1e-4 # on the order of -23 to -66

def params(data, verbose=False):
    """Computes parameters from training data.
    
    Inputs:
        - data: preprocess.Data object

    Learns a two-parameter worker model.
    pL is p(l) prior using smoothing
    pV is p(v == L | L) not using smoothing
    lCorrNeg is p(l2 | not l1) lCorrPos is p(l2 | l1)
        - uses smoothing except for diagonal

    Outputs:
        pV, pL, (lCorrPos, LCorrNeg)
    """

    posneg = data.make_posneg()

    #------------------- compute EM for single-label model ------------------
    def single_E(pV, pLabels):
        beliefs = dict()
        ll = 0
        for i in data.items:
            beliefs[i], ll_delta = inference.infer(
                dict((l, data.votes[i,l]) for
                     l in data.labels),
                data.labels,
                {'pV': pV, 'pL': pLabels},
                joint=False)

            ll += ll_delta

        # assume beta prior alpha = 1.1, beta = 1.1
        ll += np.sum(np.log(scipy.stats.beta.pdf(pLabels, 1.1, 1.1)))
        ll += np.sum(np.log(scipy.stats.beta.pdf(pV, 1.1, 1.1)))
                    
        ll = ll / len(data.items)

        return beliefs, ll

    def single_M(beliefs):
        #--- label probabilities --
        # assume beta prior alpha = 1, beta = 1
        pL = (np.sum([x[0,:] for x in beliefs.itervalues()], 0) + .1) / \
             (len(beliefs) + .2)

        #--- worker model ---
        for i in beliefs:
            assert np.isfinite(beliefs[i]).all()
        tt = (sum(np.inner(posneg[i][0,:], beliefs[i][0,:]) for 
                  i in data.items) + .1) / \
             (sum(np.inner(np.sum(posneg[i],0), beliefs[i][0,:]) for
                  i in data.items) + .2)
        
        ff = (sum(np.inner(posneg[i][1,:], beliefs[i][1,:]) for
                  i in data.items) + .1) / \
             (sum(np.inner(np.sum(posneg[i],0), beliefs[i][1,:]) for
                  i in data.items) + .2)

        pWorker = np.array([tt, ff])

        return pWorker, pL

    #--- run EM ---
    pLabels = np.zeros(len(data.labels)) + 0.2 
    pV = np.array([0.8, 0.8])
    beliefs = None
    ll = float('-inf')
    ll_change = float('inf')
    i = 0
    while i < MAX_EM_ITERS and ll_change > MIN_EM_CHANGE:
        beliefs, new_ll = single_E(pV, pLabels) 
        pV, pLabels = single_M(beliefs)
        if i == 0:
            ll_change = new_ll - ll
        else:
            ll_change = (new_ll - ll) / np.abs(ll) # make fractional improvement
        ll = new_ll
        if verbose:
            print 'll: %f (change %f)' % (ll,ll_change),pV,str(pLabels[:2])+'..'
        assert np.isfinite(ll)
        assert ll_change >= -0.00001
        i += 1

    #--- compute pL correlations ---#

    """lCorrPos(label1, label2) is P(label2 | label1)
    lCorrNeg(label1, label2) is P(label2 | not label1)

    """
    
    pos_beliefs_exp = sum(beliefs[i][0,:] for i in data.items)
    neg_beliefs_exp = sum(beliefs[i][1,:] for i in data.items)

    exppos_pos = sum(np.reshape(beliefs[i][0,:],(-1,1)) * beliefs[i][0,:] for
                     i in data.items)
    expneg_pos = sum(np.reshape(beliefs[i][1,:],(-1,1)) * beliefs[i][0,:] for
                     i in data.items)

    lCorrPos = (exppos_pos+.1) / (np.reshape(pos_beliefs_exp,(-1,1))+1.1)
    lCorrNeg = (expneg_pos+1) / (np.reshape(neg_beliefs_exp,(-1,1))+2)

    # fix diagonals
    np.fill_diagonal(lCorrPos, 1)
    np.fill_diagonal(lCorrNeg, 0)

    return {'pV':pV, 'pL':pLabels, 'lCorrPos':lCorrPos,
            'lCorrNeg':lCorrNeg}
