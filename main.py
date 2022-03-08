# from beamSearch import beamSearch
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff

from dataset import DataSet
from discretization import EqualFrequencyDiscretizer
from quality_measure import WRACC
from search_constraints import SearchConstraints
from beamSearch import beamSearch 



def read_mammals(path):
    raw_data = loadarff(path)
    df_data = pd.DataFrame(raw_data[0])
    for col, dtype in df_data.dtypes.items():
        if (dtype == np.object):  # Only process byte object columns.
            df_data[col] = df_data[col].apply(lambda x: x.decode("utf-8"))

    descriptors = df_data.iloc[:, 2:69]
    targets = df_data.iloc[:, 69:]
    ds = DataSet()
    ds.descriptors = descriptors
    ds.targets = targets.iloc[:,0]
    return ds


if __name__ == '__main__':
    ds: DataSet = read_mammals("./data/Mammals_dataset/mammals.arff")
    qm = WRACC(ds, ds.targets[ds.targets==1].index)
    quantizer = EqualFrequencyDiscretizer(10)
    search_constraints = SearchConstraints(depth=10, width=20, q=10, minimum_coverage=0.01, 
                                            quality_measure=qm, quantizer=quantizer)
    result = beamSearch(ds, qm, search_constraints)
    print(result)