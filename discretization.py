from enum import Enum
from math import floor
from sre_constants import RANGE_UNI_IGNORE
from pandas import DataFrame, Index, Series
from numpy import arange, argmin
from sklearn import mixture
from dataset import DataSet

class Discretizers(Enum):
    EQUAL_FREQUENCY = 1
    HISTOGRAM = 2

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
    
    def generate_intervals(self, column:Series, ranges: list[tuple]):
        ''' Takes values of an attribute and returns all ranges with binary discretization '''
        pass
    
    def _generate_intervals(self, column:Series, ranges: list[tuple], splits):
        intervals: list[Index] = list()
        for split in splits:
            active_ranges =  [r for r, s in zip(ranges, split) if s]
            lower = active_ranges[0][0]
            higher = active_ranges[-1][1]
            intervals.append(column.index[lower:higher])
        return [interval for interval in intervals if not interval.empty]

    def make_ranges(self, B: list):
        ranges = []
        for i in range(len(B)-1):
            b = B[i] + 1
            next_b = B[i+1]
            ranges.append((b, next_b))
        ranges[0] = (ranges[0][0] - 1, ranges[0][1])
        return ranges


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
        return self.make_ranges(B)
    def generate_intervals(self, column:Series, ranges: list[tuple]):
        splits = self.generate_all_splits(self.number_of_bins)
        return self._generate_intervals(column, ranges, splits)

    

class HistogramDiscretizer(Discretizer):
    def __init__(self, max_number_of_bins:int, dataset: DataSet):
        self.max_number_of_bins = max_number_of_bins
        self.descriptors: DataFrame = dataset.descriptors
        self.global_discretization = {}
        self.make_cuts()

    def make_cuts(self) -> None:
        for attribute_name in self.descriptors.columns:
            sorted = self.descriptors[attribute_name].loc[ 
                self.descriptors.index[
                    self.descriptors[attribute_name].argsort()
                    ]
            ]
            X = sorted.values.reshape(-1,1)
            N = arange(1, self.max_number_of_bins +1)
            models = [None for i in range(len(N))]
            for i in range(len(N)):
                models[i] = mixture.GaussianMixture(N[i]).fit(X)
            AIC = [m.aic(X) for m in models]
            gmm = models[argmin(AIC)]
            labels = Series(gmm.predict(X), index=sorted.index)
            B = [-1]
            for l in labels.unique():
                B.append(B[-1] + sum(labels==l))
            B[0] = 0
            self.global_discretization[attribute_name] = self.make_ranges(B)

    def discretize(self, column: Series):
        discretization = []
        for range in self.global_discretization[column.name]:
            if range[1] < len(column):
                discretization.append(range)
            elif range[0] < len(column):
                # Cut last bin to match column size
                discretization.append((range[0], len(column) - 1))
            else:
                # Ignore all other ranges
                break
        return discretization

    def generate_intervals(self, column: Series, ranges: list[tuple]):
        splits = self.generate_all_splits(len(ranges) - 1)
        return self._generate_intervals(column, ranges, splits)