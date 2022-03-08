from math import floor
from pandas import Series
from typing import List

class Discretizer:
    def discretize():
        ''' Takes a column and makes number_of_bins many bins, returns pandas.Index of each bin separately '''
        pass


# use only geq
class EqualFrequencyDiscretizer(Discretizer):
    def __init__(self, number_of_bins:int):
        self.number_of_bins = number_of_bins
    def discretize(self, column:Series):
        pass
        # n:int = len(column)

        # B = list()
        # for b in range(1, self.number_of_bins + 1):
        #     ix = floor(n*b/self.number_of_bins)
        #     B.append(column[ix])
        # return B