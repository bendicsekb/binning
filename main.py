# from beamSearch import beamSearch
from enum import Enum
import string
import time
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff

from dataset import DataSet, DataSets
from discretization import Discretizer, Discretizers, EqualFrequencyDiscretizer, HistogramDiscretizer
from quality_measure import WRACC, ZSCORE
from search_constraints import SearchConstraints
from beamSearch import beamSearch, modifiedBeamSearch 

import pickle
import json
import joblib

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
    ds.targets = targets['Apodemus_sylvaticus']
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

class RunType(Enum):
    NORMAL = 1
    SAME_DESCRIPTORS = 2


def main(dataset_name: DataSets, quantizer_type: Discretizers, eqf_bins: int, run_type: RunType, descriptor_set: list[str]=[]):
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
    
    quantizer = Discretizer()
    match quantizer_type:
        case Discretizers.EQUAL_FREQUENCY:
            quantizer = EqualFrequencyDiscretizer(eqf_bins)
        case Discretizers.HISTOGRAM:
            quantizer = HistogramDiscretizer(10, ds)

    result = []
    match run_type:
        case RunType.NORMAL:
            search_constraints = SearchConstraints(depth=10, width=20, q=10, minimum_coverage=0.01, 
                                                    quality_measure=qm, quantizer=quantizer, use_columns=list(ds.descriptors.columns))
            result = beamSearch(ds, qm, search_constraints)
            result.sort()
            
        case RunType.SAME_DESCRIPTORS:
            search_constraints = SearchConstraints(depth=10, width=20, q=10, minimum_coverage=0.01, 
                                                    quality_measure=qm, quantizer=quantizer, use_columns=list())
            result = modifiedBeamSearch(ds, qm, search_constraints, descriptor_set)

    return result, ds


def get_descriptor_set(resultset_filename) -> list[str]:
    resultset = []
    with open(resultset_filename, 'rb') as f:
        raw_resultset = pickle.load(f)
    for res in raw_resultset:
        resultset.append(res[2])

    descriptions = []
    for res in resultset:
        description_set = list(dict.fromkeys([d.attribute for d in res.description]).keys())
        descriptions.append(description_set)

    return descriptions

def resultset_filename(base_filename):
    return f'save/dump/{base_filename}_resultset.pickle'

def dataset_filename(base_filename):
    return f'save/dump/{base_filename}_dataset.pickle'

def discretizations_filename(base_filename):
    return f'save/json/{base_filename}_discretizations.json'

def dump(result, dataset, base_filename, discretizations):
    with open(resultset_filename(base_filename), 'wb') as f:
        pickle.dump(result, f)
    with open(dataset_filename(base_filename), 'wb') as f:
        pickle.dump(dataset, f)
    with open(discretizations_filename(base_filename), 'w') as f:
        print(json.dumps(discretizations, indent=4), file=f)


def process_result(result, dataset, base_filename):
    resultset = [res for _, _, res in result]
    discretizations = [res.discretization_boundaries for res in resultset]
    dump(result, dataset, base_filename, discretizations)
    

def process_copy_result(result, dataset, base_filename):
    resultset = []
    for res in result:
        resultset.append(res[1][0][2])
    
    discretizations = [res.discretization_boundaries for res in resultset]
    dump(result, dataset, base_filename, discretizations)

def run_eqf(dataset_type, eqf_bins):
    discretizer_type = Discretizers.EQUAL_FREQUENCY
    t = time.time()
    res = []
    ds = DataSet()
    res, ds = main(dataset_type, discretizer_type, eqf_bins, RunType.NORMAL)
    file_base = f'{dataset_type}_{discretizer_type}_{eqf_bins}bins'
    process_result(res, ds, file_base)
    t_eq = time.time() - t
    print(f'{file_base} took {round(t_eq, 2)} seconds')

def run_copy_hist(dataset_type: DataSets, eqf_bins: int):
    discretizer_type = Discretizers.HISTOGRAM
    res = []
    ds = DataSet()
    eqf_file_base = f'{dataset_type}_{Discretizers.EQUAL_FREQUENCY}_{eqf_bins}bins'
    descriptor_set = get_descriptor_set(resultset_filename(eqf_file_base))
    res, ds = main(dataset_type, discretizer_type, eqf_bins, RunType.SAME_DESCRIPTORS, descriptor_set)
    file_base = f'{dataset_type}_{discretizer_type}_{eqf_bins}bins'
    process_copy_result(res, ds, file_base)
    return res
    

def run_discretizers():
    for dataset_type in [DataSets.IONOSPHERE, DataSets.MAMMALS]:
        # joblib.Parallel(n_jobs=10)(joblib.delayed(run_eqf)(dataset_type, eqf_bins) for eqf_bins in [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])            
        joblib.Parallel(n_jobs=10)(joblib.delayed(run_copy_hist)(dataset_type, eqf_bins) for eqf_bins in [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])            
    
    t = time.time()
    if False:
        res, ds = main(DataSets.YEARPREDICTIONMSD, Discretizers.HISTOGRAM)
        process_result(res, ds, 'YPD_aa_max10bins_HIST')
    t_hist = time.time() - t
    print(f'Histogram: {round(t_hist, 2)}s')


if __name__ == '__main__':
    profiling = False
    if profiling:
        import cProfile
        cProfile.run('run_discretizers()', './profiling/ionosphere_03_17.dat')
    else:
        run_discretizers()
