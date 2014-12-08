"""run.py

Script to load a .csv file with votes, infer labels using both the "independent" and "joint" models,
and print taxonomies.
"""

import sys
import collections
import numpy as np
import inference
import learning
import preprocess
import taxonomy

def make_labels_to_items(items_to_labels):
    labels_to_items = collections.defaultdict(list)
    for i in items_to_labels:
        for l in items_to_labels[i]:
            labels_to_items[l].append(i)
    return labels_to_items

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'usage: {} filename.csv'.format(sys.argv[0])
        sys.exit()

    data = preprocess.Data(sys.argv[1])
    params = learning.params(data)

    items_to_labels = {'independent': dict(),
                       'joint': dict()}
    for i in data.items:
        votes_item = dict((l, data.votes[i, l]) for l in data.labels)
        probabilities = {'independent': inference.infer(votes_item, data.labels, params, joint=False)[0],
                         'joint': inference.infer(votes_item, data.labels, params, joint=True)[0]}

        for s in ['independent', 'joint']:
            x = np.round(probabilities[s][0])
            items_to_labels[s][i] = [data.labels[ind] for ind in np.where(x==1)[0]]

    labels_to_items = dict()
    for s in ['independent', 'joint']:
        labels_to_items[s] = make_labels_to_items(items_to_labels[s])
        print '---------Taxonomy for {} model------------'.format(s)
        print taxonomy.make_taxonomy(labels_to_items[s])
