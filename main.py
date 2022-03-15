# from beamSearch import beamSearch
import string
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff

from dataset import DataSet, DataSets
from discretization import EqualFrequencyDiscretizer
from quality_measure import WRACC
from search_constraints import SearchConstraints
from beamSearch import beamSearch 



def read_mammals(path: string):
    raw_data = loadarff(path)
    df_data = pd.DataFrame(raw_data[0])
    for col, dtype in df_data.dtypes.items():
        if (dtype == object):  # Only process byte object columns.
            df_data[col] = df_data[col].apply(lambda x: int(x.decode("utf-8")))

    descriptors = df_data.iloc[:, 2:69]
    targets = df_data.iloc[:, 69:]
    ds = DataSet()
    ds.descriptors = descriptors
    ds.targets = targets.iloc[:,0]
    return ds

def read_ionosphere(path: string):
    columns = [f'd{i}' for i in range(1, 35)]
    columns.append('t')
    with open(path) as datafile:
        df = pd.read_csv(datafile, header=None, names=columns )
    df['t'] = df['t'].apply(lambda elem: 1 if elem == 'g' else 0 )
    ds = DataSet()
    ds.descriptors = df.drop('t', axis=1)
    ds.targets = df['t']
    return ds

def main():
    ds: DataSet = DataSet()
    dataset_name = DataSets.IONOSPHERE

    match dataset_name:
        case DataSets.IONOSPHERE:
            ds: DataSet = read_ionosphere("./data/Ionosphere/ionosphere.data")
        case DataSets.MAMMALS:
            ds: DataSet = read_mammals("./data/Mammals_dataset/mammals.arff")
    
    qm = WRACC(ds, ds.targets[ds.targets==1].index)
    quantizer = EqualFrequencyDiscretizer(10)
    search_constraints = SearchConstraints(depth=10, width=20, q=10, minimum_coverage=0.01, 
                                            quality_measure=qm, quantizer=quantizer)
    result = beamSearch(ds, qm, search_constraints)
    print(result)

if __name__ == '__main__':
    import cProfile
    cProfile.run('main()', './profiling/ionosphere_03_15.dat')
