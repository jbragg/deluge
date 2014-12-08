"""taxonomy.py"""

from __future__ import division
import itertools

def fraction_of_in(l1,l2):
    return len(set(l1).intersection(set(l2))) / len(set(l1))

def make_taxonomy(labels_to_items, verbose=False):
    # remove categories with fewer than two tips
    tax = dict((k,v) for k,v in labels_to_items.items() if len(v) > 1)

    # categories in descending length order
    tax = sorted(tax.items(), key=lambda x: len(x[1]), reverse=True)

    def recurse(tax,parents=[]):
        if len(tax) == 0:
            return []

        largest_category, largest_items = tax[0]

        assert len(set(largest_items)) == len(largest_items)

        # remove categories that share over 75% tips with first category
        potential = [(k,v) for k,v in tax[1:] \
            if not (fraction_of_in(v,largest_items) > .75 \
                and fraction_of_in(largest_items,v) > .75)]

        # potential nested categories
        nested = [(k,v) for k,v in potential \
                            if fraction_of_in(v,largest_items) > .75]
        others = [x for x in potential if x not in nested]

        if verbose:
            print 
            print
            print 'considering %s' % largest_category
            print 'potential nested: '
            print [k for k,v in nested]
            print 'others: '
            print [(k, fraction_of_in(v,largest_items)) for k,v in others]

        return [{"members": largest_items, "parents": parents, "label": largest_category}] + recurse(nested,parents=parents+[largest_category]) + recurse(others,parents=parents)

    result = recurse(tax)
    all_items = set(list(itertools.chain.from_iterable([labels_to_items[k] for k in labels_to_items])))
    categorized = set(list(itertools.chain.from_iterable([d['members'] for d in result])))
    other_items = [i for i in all_items if i not in categorized]

    return result+[{"members": other_items, "parents": [], "label": 'other'}]
