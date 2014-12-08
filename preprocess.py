"""preprocess.py"""

from __future__ import division
import csv
import collections
import numpy as np
import itertools

def load_data(fname):
    """Reads a csv file and loads votes and labels into convenient format.
    
    Input is csv file with the following fields:
    - item: item name (str)
    - label: label name (str)
    - selected: vote (int)

    Outputs:
    - items: item names (list: str)
    - labels: label names (list: str)
    - votes: (item name, label name) -> votes (list: bool)
    """

    with open(fname, 'r') as f:
        reader = csv.DictReader(f)
        items = set()
        labels = set()
        votes = collections.defaultdict(list)
        for row in reader:
            items.add(row['item'])
            labels.add(row['label'])
            votes[row['item'], row['label']].append(bool(int(row['selected'])))

        return list(items), list(labels), votes

class Data:
    def __init__(self, fin):
        items, labels, votes = load_data(fin)
        self.items = items
        self.labels = labels
        self.votes = votes
            
    def make_posneg(self):
        """Convert votes to matrix format.
        
        Output: item name -> matrix (2 x |labels|) with the number of
        positive votes (first row) and negative votes (second row)
        """
        posneg = dict()
        for i in self.items:
            pos = np.array([sum(self.votes[i,l]) for
                            l in self.labels])
            neg = np.array([len(self.votes[i,l]) for
                            l in self.labels]) - pos
            posneg[i] = np.vstack((pos, neg))

        return posneg

    def __repr__(self):
        return '%s, %s\nitem1: %s' % (self.items, self.labels, self.votes.items()[0])
