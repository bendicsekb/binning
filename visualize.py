#%%
from cProfile import label
import dataset
import discretization
import main
import matplotlib.pyplot as plt
import json
import pickle

#%%

ionosphere_dataset: dataset.DataSet = main.read_ionosphere("./data/Ionosphere/ionosphere.data")
mammals_dataset: dataset.DataSet = main.read_mammals("./data/Mammals_dataset/mammals.arff")
year_prediction_MSD_dataset : dataset.DataSet = main.read_YearPredictionMSD('./data/YearPredictionMSD/year_prediction.csv')






# %%
ds = mammals_dataset

for descriptor_name in ds.descriptors.columns:
    column = ds.descriptors[descriptor_name]
    fig, ax = plt.subplots(3, 1)
    ax[0].hist(column)
    ax[0].set_title(descriptor_name)
    fig.show()




# %%
with open('./save/json/Mammals_EQF_discretizations.json', 'r') as f:
    eqf_discretizations = json.load(f)
# %%
eqf_discretizations[0]['max_temp_feb_utm']
# %%

def find_discretizations(all_discretizations, descriptor_name):
    ret = []
    for discretizations in all_discretizations:
        if descriptor_name in discretizations:
            for disc in discretizations[descriptor_name]:
                ret.append(disc[1])
    return ret


# %%
found = find_discretizations(eqf_discretizations, 'max_temp_feb_utm')



# %%
len(found)
# %%
for f in found:
    print(f)



# %%
mammals_dataset: dataset.DataSet = main.read_mammals("./data/Mammals_dataset/mammals.arff")


mammals_resultset_eqf = []
with open('./save/dump/Mammals_EQF_resultset.pickle', 'rb') as f:
    results = pickle.load(f)
for res in results:
    mammals_resultset_eqf.append(res[2])

histogram_quantizer = discretization.HistogramDiscretizer(10, mammals_dataset)

# %%
for i in range(len(mammals_resultset_eqf)):
    subgroup_eqf = mammals_resultset_eqf[i]
    discretization_boundaries = subgroup_eqf.discretization_boundaries
    for attribute_name, boundaries in discretization_boundaries.items():
        column = mammals_dataset.descriptors[attribute_name]
        fig, ax = plt.subplots()
        ax.hist(column)
        level = 0
        first = True
        for _, boundary in boundaries:
            bounds = [boundary[0][0]] + [b[1] for b in boundary]
            for bound in bounds:
                if first: 
                    ax.axvline(bound, color='r', label='EQF', ymin=0.9+(level*0.05), ymax=0.999)
                    first = False
                else:
                    ax.axvline(bound, color='r', ymin=0.9+(level*0.05), ymax=0.999)
            level += 1

        sorted = mammals_dataset.descriptors.loc[mammals_dataset.descriptors.index[mammals_dataset.descriptors[attribute_name].argsort()]]
        splits: list[tuple] = histogram_quantizer.discretize(sorted[attribute_name])
        boundary = [(sorted[attribute_name][sorted.index[split[0]]], sorted[attribute_name][sorted.index[split[1]]]) for split in splits]
        bounds = [boundary[0][0]] + [b[1] for b in boundary]
        first = True
        for bound in bounds:
                if first: 
                    ax.axvline(bound, color='b', label='HIST', ymin=0.001, ymax=0.1)
                    first = False
                else:
                    ax.axvline(bound, color='b', ymin=0.001, ymax=0.1)

        ax.legend(loc=0)
        ax.set_title(attribute_name)
        fig.savefig(f'./save/images/{i}_{attribute_name}.png', facecolor='white')





#%%







# %%
attribute_name = 'max_temp_feb_utm'
sorted = mammals_dataset.descriptors.loc[mammals_dataset.descriptors.index[mammals_dataset.descriptors[attribute_name].argsort()]]
splits: list[tuple] = histogram_quantizer.discretize(sorted[attribute_name])
boundary = [(sorted[attribute_name][sorted.index[split[0]]], sorted[attribute_name][sorted.index[split[1]]]) for split in splits]
bounds = [boundary[0][0]] + [b[1] for b in boundary]

# %%
bounds
# %%
