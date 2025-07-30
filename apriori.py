import itertools
from collections import defaultdict


def apriori(transactions, min_support=0.5, max_length=None):
    """
    Returns: dict mapping frozenset(itemset) -> support (float)
    """
    if len(transactions) == 0:
        raise ValueError()
    
    itemsets = defaultdict(int)
    num_transactions = len(transactions)
    items = set(itertools.chain.from_iterable(transactions))

    k = 1
    current_itemsets = [frozenset([item]) for item in items]
    while current_itemsets and (max_length is None or k <= max_length):

        counts = defaultdict(int)
        for t in transactions:
            t_set = set(t)
            for itemset in current_itemsets:
                if itemset.issubset(t_set):
                    counts[itemset] += 1

        current_itemsets = [itemset for itemset, count in counts.items() if count / num_transactions >= min_support]
        
        for itemset in current_itemsets:
            itemsets[itemset] = counts[itemset]

        k += 1
        current_itemsets = list(set([i.union(j) for i in current_itemsets for j in current_itemsets if len(i.union(j)) == k]))

    return {itemset: count / num_transactions for itemset, count in itemsets.items()}


transactions = [
    {'bread', 'milk'},
    {'bread', 'diaper', 'beer', 'eggs'},
    {'milk', 'diaper', 'beer', 'cola'},
    {'bread', 'milk', 'diaper', 'beer'},
    {'bread', 'milk', 'diaper', 'cola'}
]
result = apriori(transactions, min_support=0.6)
for k in sorted(result, key=lambda x: (len(x), sorted(x))):
    print(sorted(list(k)), round(result[k], 2))