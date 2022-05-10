#%% 
from cProfile import label
from enum import Enum
import os
from dataset import DataSets
import discretization
import main
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import json
import pickle
import pandas as pd
import numpy as np



#%%
raw = pd.read_csv("save/runtimes/runtimes.csv")
ionosphere = raw[raw["dataset"]== "DataSets.IONOSPHERE"].drop(["dataset", "run"], axis=1)
ionosphere_means = ionosphere.groupby(["bins"]).mean()
mammals = raw[raw["dataset"]== "DataSets.MAMMALS"].drop(["dataset", "run"], axis=1)
mammals_means = mammals.groupby(["bins"]).mean()
# %%

# %%
def plot_runtimes(means, path, dataset_name):
    fig, ax = plt.subplots(figsize=(4, 6))
    plt.rcParams.update({'mathtext.default':  'regular' })

    x = means.index
    eqf = means["eqf"]
    hist = means["hist"]
    copy_hist = means["copy_hist"]

    ax.plot(eqf, "b^", fillstyle="none", label='EQF')
    eqf_poly_params = np.polyfit(x, eqf, 2)
    eqf_poly = np.poly1d(eqf_poly_params)
    ax.plot(x, eqf_poly(x), "k:", lw=0.5)

    ax.plot(hist, "go", fillstyle="none", label='HIST')
    hist_poly_params = np.polyfit(x, hist, 1)
    hist_poly = np.poly1d(hist_poly_params)
    ax.plot(x, hist_poly(x), "k:", lw=0.5)

    ax.plot(copy_hist, "r+", label='$HIST_{Copy}$')
    copy_hist_poly_params = np.polyfit(x, copy_hist, 1)
    copy_hist_poly = np.poly1d(copy_hist_poly_params)
    ax.plot(x, copy_hist_poly(x), "k:", lw=0.5)

    fig.suptitle(dataset_name)
    ax.legend()
    plt.xlabel('Number of bins')
    plt.ylabel('Runtime (s)')

    plt.savefig(path, facecolor="white")

# %%
plot_runtimes(ionosphere_means, "save/runtimes/ionosphere_runtimes.png", "Ionosphere")
plot_runtimes(mammals_means, "save/runtimes/mammals_runtimes.png", "Mammals")

# %%
# %%
