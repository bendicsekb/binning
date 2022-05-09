from cProfile import label
from enum import Enum
import os
import dataset
import discretization
import main
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import json
import pickle

def read_discretizations(file_base, type_of_file, disc_type, ds_type, eqf_bins, hist_maxbins, copy):

    if(disc_type == discretization.Discretizers.EQUAL_FREQUENCY):
        boundary_filename = f'{file_base}{ds_type}_{disc_type}_{eqf_bins}bins_{type_of_file}'
    elif(disc_type == discretization.Discretizers.HISTOGRAM):
        boundary_filename = f'{file_base}{ds_type}_{disc_type}_{f"COPY_{eqf_bins}bins_" if copy else ""}max{hist_maxbins}bins_{type_of_file}'
    else:
        boundary_filename = 'No such discretizer'
    with open(boundary_filename, 'rb') as f:
        discretizations = json.load(f)
    return discretizations


def read_dump(file_base, type_of_file, disc_type, ds_type, eqf_bins, hist_maxbins, copy):
    if(disc_type == discretization.Discretizers.EQUAL_FREQUENCY):
        boundary_filename = f'{file_base}{ds_type}_{disc_type}_{eqf_bins}bins_{type_of_file}'
    elif(disc_type == discretization.Discretizers.HISTOGRAM):
        boundary_filename = f'{file_base}{ds_type}_{disc_type}_{f"COPY_{eqf_bins}bins_" if copy else ""}max{hist_maxbins}bins_{type_of_file}'
    else:
        boundary_filename = 'No such discretizer'
    with open(boundary_filename, 'rb') as f:
        dump = pickle.load(f)
    return dump


class Grouping():
    def __init__(self, discretizations, resultset, ds) -> None:
        self.discretizations = discretizations
        self.resultset = resultset
        self.dataset = ds


class TestType(Enum):
    EQF = 1
    HISTOGRAM = 2
    COPY_HISTOGRAM = 3


def keep_only_best(hist_resultset):
    ret = []
    for res_set in hist_resultset:
        res_set[1].sort()
        ret.append(res_set[1][-1])

    return ret


def order_intervals(intervals):
    ordered = {}
    for k, interval in intervals.items():
        ordered[k] = []
        for cond in interval:
            match cond.negated:
                case False:
                    ordered[k].insert(0, cond.value) # prepend
                case True: 
                    ordered[k].append(cond.value)
    return ordered


def extract_intervals(descr):
    intervals = {}
    for cond in descr:
        if cond.attribute in intervals:
            intervals[cond.attribute].append(cond)
        else:
            intervals[cond.attribute] = [cond]

    return order_intervals(intervals)


def plot_eq_hist(fig, ax, k, d_eqf, d_hist, column):
    palette = ['r', 'm', 'y', 'b']
    ax.hist(column, bins=25)
    level = 0
    first = True
    for _, boundary in d_eqf:
        bounds = [boundary[0][0]] + [b[1] for b in boundary]
        for bound in bounds:
            if first: 
                ax.axvline(bound, color=palette[level], label=f'EQF', ymin=0.8+(level*0.05), ymax=0.999)
                first = False
            else:
                ax.axvline(bound, color=palette[level], ymin=0.8+(level*0.05), ymax=0.999)
        level += 1


    level = 0
    first = True
    for _, boundary in d_hist:
        bounds = [boundary[0][0]] + [b[1] for b in boundary]
        for bound in bounds:
            if first: 
                ax.axvline(bound, color='b', label='HIST', ymin=0.001, ymax=0.2)
                first = False
            else:
                ax.axvline(bound, color='b', ymin=0.001, ymax=0.2)
        level += 1

    ax.legend()
    ax.set_title(f'{k}')
    return fig, ax

def plot_subgroup(fig, ax, interval, is_eqf, facecolor):
    bottom, top = ax.get_ylim()
    if is_eqf:
        o = (interval[0], top-10)
    else:
        o = (interval[0], bottom)
    w = interval[1] - interval[0]
    h = 10
    rec = patches.Rectangle(o, w, h, facecolor = facecolor, alpha=0.6)
    ax.add_patch(rec)
    
    return fig, ax


def plot_results(fig, ax, eqf_res, hist_res, d_eqf, d_hist, ds):

    eqf_intervals = extract_intervals(eqf_res[2].description)
    hist_intervals = extract_intervals(hist_res[2].description)

    keys = d_eqf.keys()
    for i, k in enumerate(keys):
        fig, ax[i] = plot_eq_hist(fig, ax[i], k, d_eqf[k], d_hist[k], ds.descriptors[k])
        fig, ax[i] = plot_subgroup(fig, ax[i], eqf_intervals[k], is_eqf=True, facecolor='g')
        fig, ax[i] = plot_subgroup(fig, ax[i], hist_intervals[k], is_eqf=False, facecolor='m')
        
    fig.suptitle(f'EQF: {round(eqf_res[0], 5)} vs HIST: {round(hist_res[0], 5)}')
    return fig, ax


if __name__=='__main__':
    for ds_type in [dataset.DataSets.IONOSPHERE, dataset.DataSets.MAMMALS]:
        for eqf_bins in [5,10,15,20,25]:
            hist_maxbins = 20

            folder_name = f'save/images/copy/{ds_type}/eqf_{eqf_bins}_bins'
            if(not os.path.exists(folder_name)):
                os.makedirs(folder_name)


            test_result = {}
            for test_type in [TestType.EQF, TestType.COPY_HISTOGRAM]: # [TestType.EQF, TestType.HISTOGRAM, TestType.COPY_HISTOGRAM]:
                match test_type:
                    case TestType.EQF:
                        disc_type = discretization.Discretizers.EQUAL_FREQUENCY
                        copy = False
                    case TestType.HISTOGRAM:
                        disc_type = discretization.Discretizers.HISTOGRAM
                        copy = False
                    case TestType.COPY_HISTOGRAM:
                        disc_type = discretization.Discretizers.HISTOGRAM
                        copy = True
                        hist_maxbins = 20  # fixed for COPY_HISTOGRAM

                type_of_file = 'discretizations.json'
                file_base = f'save/json/new/discretization/'
                discr = read_discretizations(
                    file_base, type_of_file, disc_type, ds_type, eqf_bins, hist_maxbins, copy)

                type_of_file = 'resultset.pickle'
                file_base = f'save/dump/new/resultset/'
                resultset = read_dump(file_base, type_of_file, disc_type,
                                    ds_type, eqf_bins, hist_maxbins, copy)

                file_base = f'save/dump/new/dataset/'
                type_of_file = 'dataset.pickle'
                ds = read_dump(file_base, type_of_file, disc_type,
                            ds_type, eqf_bins, hist_maxbins, copy)

                group = Grouping(discr, resultset, ds)
                test_result[test_type] = group


            ds = test_result[TestType.EQF].dataset
            eqf_resultset = test_result[TestType.EQF].resultset
            hist_resultset = keep_only_best(test_result[TestType.COPY_HISTOGRAM].resultset)

            for eqf_res, hist_res in zip(eqf_resultset, hist_resultset):
                d_eqf = eqf_res[2].discretization_boundaries
                d_hist = hist_res[2].discretization_boundaries
                plt.figure()
                keys = d_eqf.keys()
                fig, ax = plt.subplots(ncols=len(keys), figsize=(20,5))
                fig, ax = plot_results(fig, ax, eqf_res, hist_res, d_eqf, d_hist, ds)
                ratio = hist_res[0] / eqf_res[0]
                file_name = f'{folder_name}/{str(round(ratio, 3)).replace(".", "_")}__{"_".join(keys)}'
                plt.savefig(file_name)
                plt.clf()
                plt.cla()
                plt.close('all')

