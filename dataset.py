from enum import Enum
from pandas import DataFrame

class DataSet:
    descriptors: DataFrame
    targets: DataFrame


class DataSets(Enum):
    IONOSPHERE = 1
    MAMMALS = 2
    YEARPREDICTIONMSD=3

    