from math import floor
from pandas import Index, Series
from typing import List

class Discretizer:
    def discretize():
        ''' Takes a column and makes number_of_bins many bins, returns pandas.Index of each bin separately '''
        pass
    def generate_all_splits(self, n):
        ranges = []
        for i in range(1,n):
            arr = [True] * i 
            arr.extend([False] * (n-i))
            ranges.append(arr)
        neg_ranges = [[not elem for elem in r] for r in ranges]
        splits = []
        splits.extend(ranges)
        splits.extend(neg_ranges)
        for j in range(1, len(ranges)):
            for i, a in enumerate(ranges[j:]):
                b = neg_ranges[i]
                splits.append([ai and bi for ai, bi in zip(a, b)])
        return splits
        
    def generate_splits(self, column:Series, ranges: list[tuple]):
        splits = self.generate_all_splits(self.number_of_bins)
        intervals: list[Index] = list()
        for split in splits:
            idx = Index([])
            for lower, higher in [r for r, s in zip(ranges, split) if s]:
                idx = idx.append(column.index[lower:higher])
            intervals.append(idx)
        return intervals
    


# use only geq
class EqualFrequencyDiscretizer(Discretizer):
    def __init__(self, number_of_bins:int):
        self.number_of_bins = number_of_bins
    def discretize(self, column:Series):
        n:int = len(column) -1 

        B = [0]
        for b in range(1, self.number_of_bins):
            ix = floor(n*b/(self.number_of_bins))
            B.append(ix)
        B.append(n)
        ranges = []
        for i in range(len(B)-1):
            b = B[i]
            next_b = B[i+1]
            ranges.append((b, next_b))
        return ranges

