from ast import Index, match_case
from cmath import sqrt
from enum import Enum
from dataset import DataSet
from pandas import Index
from subgroup import SubGroup

class QualityMeasureType(Enum):
    WRACC = 1
    ZSCORE = 2

class QualityMeasure:
    def __init__(self, ds: DataSet, y: Index) -> None:
        self.ds = ds
        self.y = y
    def run_on(self, subgroup: SubGroup) -> float:
        pass

class WRACC(QualityMeasure):
    def __init__(self, ds: DataSet, y: Index) -> None:
        super().__init__(ds, y)
        self.num_ds_rows = len(self.ds.targets.index)
        self.target_coverage = len(self.y) / self.num_ds_rows

    def run_on(self, subgroup: SubGroup) -> float:
        y_hat = subgroup.targets.index
        num_tp = len(self.y.intersection(y_hat))
        tp = num_tp / self.num_ds_rows
        subgroup_coverage = len(y_hat) / self.num_ds_rows

        return tp - (subgroup_coverage * self.target_coverage)

class ZSCORE(QualityMeasure):
    def __init__(self, ds: DataSet, y: Index) -> None:
        super().__init__(ds, y)
        # TODO This only works for single target
        single_target = ds.targets[y].iloc[:,0]
        self.mean = single_target.mean() 
        self.std = single_target.std()
    def run_on(self, subgroup: SubGroup) -> float:
        y_hat = subgroup.targets
        n = len(y_hat.index)
        # TODO This only works for single target
        subgroup_mean = y_hat.iloc[:,0].mean()
        return (sqrt(n) * (subgroup_mean - self.mean)) / self.std