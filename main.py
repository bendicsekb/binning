# from beamSearch import beamSearch
import string
import time
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff

from dataset import DataSet, DataSets
from discretization import Discretizer, Discretizers, EqualFrequencyDiscretizer, HistogramDiscretizer
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
    not_used = ['drop1', 'drop2']
    columns = not_used + [f'd{i}' for i in range(1, 33)]
    columns.append('t')
    with open(path) as datafile:
        df = pd.read_csv(datafile, header=None, names=columns )
    df = df.drop(not_used, axis=1)
    df['t'] = df['t'].apply(lambda elem: 1 if elem == 'g' else 0 )
    ds = DataSet()
    ds.descriptors = df.drop('t', axis=1)
    ds.targets = df['t']
    return ds

def read_YearPredictionMSD(path:string):
    df = pd.read_csv(path)
    ds = DataSet()
    ds.descriptors = df.drop('label', axis=1)
    ds.targets = df['label']
    return ds



def main(dataset_name: DataSets, quantizer_type: Discretizers):
    ds: DataSet = DataSet()

    match dataset_name:
        case DataSets.IONOSPHERE:
            ds: DataSet = read_ionosphere("./data/Ionosphere/ionosphere.data")
            qm = WRACC(ds, ds.targets[ds.targets==1].index)
        case DataSets.MAMMALS:
            ds: DataSet = read_mammals("./data/Mammals_dataset/mammals.arff")
    qm = WRACC(ds, ds.targets[ds.targets==1].index)
        case DataSets.YEARPREDICTIONMSD:
            ds : DataSet = read_YearPredictionMSD('./data/YearPredictionMSD/year_prediction.csv')
            qm = WRACC(ds, ds.targets[ds.targets==2009].index)
    
    # qm = WRACC(ds, ds.targets[ds.targets==1].index)

    quantizer = Discretizer()
    match quantizer_type:
        case Discretizers.EQUAL_FREQUENCY:
            quantizer = EqualFrequencyDiscretizer(10)
        case Discretizers.HISTOGRAM:
            quantizer = HistogramDiscretizer(30, ds)

    search_constraints = SearchConstraints(depth=10, width=20, q=10, minimum_coverage=0.01, 
                                            quality_measure=qm, quantizer=quantizer)

    result = beamSearch(ds, qm, search_constraints)
    for res in result:
        print(res[2])

def run_discretizers():
    t = time.time()
    if True:
        main(DataSets.MAMMALS, Discretizers.EQUAL_FREQUENCY)
    t_eq = time.time() - t
    t = time.time()
    if True:
        main(DataSets.MAMMALS, Discretizers.HISTOGRAM)
    t_hist = time.time() - t
    print(f'Equal Frequency: {round(t_eq, 2)}s \tHistogram: {round(t_hist, 2)}s')
if __name__ == '__main__':
    profiling = False
    if profiling:
        import cProfile
        cProfile.run('run_discretizers()', './profiling/ionosphere_03_17.dat')
    else:
        run_discretizers()
