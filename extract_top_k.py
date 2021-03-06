# %%
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


# %%
raw = pd.read_csv("save/perf/results.csv")
ionosphere = raw[raw["dataset"] == "DataSets.IONOSPHERE"].drop(
    ["dataset", "run"], axis=1)
mammals = raw[raw["dataset"] == "DataSets.MAMMALS"].drop(
    ["dataset", "run"], axis=1)

# %%
flat = pd.DataFrame(columns=["dataset", "discr",
                    "bins", "quality", "description"])
idx = 0

for ds_name, ds in zip(["i", "m"], [ionosphere, mammals]):
    for ix, record in ds.iterrows():
        num_b = record["bins"]

        for discr in ["eqf", "hist"]:
            elem = record[discr]

            parts = elem.strip("[]").split("), (")
            parts = [part.strip("()") for part in parts]

            for part in parts:
                flat.loc[idx] = [ds_name, discr, num_b, round(float(part.split(",")[0]), 5), part.split("[")[1].strip("[]")]
                idx += 1


# %%
for ds_name, ds in zip(["i", "m"], [flat[flat["dataset"] == "i"], flat[flat["dataset"] == "m"]]):

    for discr in ["eqf", "hist"]:
        rec = ds[ds["discr"] == discr].drop_duplicates()
        rec_5bins = rec[rec["bins"] == 5].sort_values(by="quality", ascending=False).head(5)
        rec_10bins = rec[rec["bins"] == 10].sort_values(by="quality", ascending=False).head(5)
        rec_15bins = rec[rec["bins"] == 15].sort_values(by="quality", ascending=False).head(5)
        rec_20bins = rec[rec["bins"] == 20].sort_values(by="quality", ascending=False).head(5)
        rec_25bins = rec[rec["bins"] == 25].sort_values(by="quality", ascending=False).head(5)

        rec_5bins.to_csv(f"save/perf/{ds_name}_{discr}_5_bins.csv")
        rec_10bins.to_csv(f"save/perf/{ds_name}_{discr}_10_bins.csv")
        rec_15bins.to_csv(f"save/perf/{ds_name}_{discr}_15_bins.csv")
        rec_20bins.to_csv(f"save/perf/{ds_name}_{discr}_20_bins.csv")
        rec_25bins.to_csv(f"save/perf/{ds_name}_{discr}_25_bins.csv")

        print(f"{ds_name}\t{discr}\t{5}bins\t{round(rec_5bins.mean()['quality'], 5)}")
        print(f"{ds_name}\t{discr}\t{10}bins\t{round(rec_10bins.mean()['quality'], 5)}")
        print(f"{ds_name}\t{discr}\t{15}bins\t{round(rec_15bins.mean()['quality'], 5)}")
        print(f"{ds_name}\t{discr}\t{20}bins\t{round(rec_20bins.mean()['quality'], 5)}")
        print(f"{ds_name}\t{discr}\t{25}bins\t{round(rec_25bins.mean()['quality'], 5)}")

# %%


# %%


# %%

record = ionosphere.loc[0]
elem = record["copy_hist"]
elem
# [
#     (
#         ['d3', 'd25', 'd5'],
#         [
#             (0.12367594418876465, 200000000000000010732324408786304, Quality: 0.12367594418876465,
#             [d3 > 0.0, d3 <= 0.99352, d25 > -0.70352,
#                 d25 <= 0.9212, d5 > 0.03759, d5 <= 0.95691]
#             ),
#             (0.1218496603111987, 200000000000000010732324408786305, Quality: 0.1218496603111987,
#             [d3 > 0.0, d3 <= 0.99352, d25 > -0.70352,
#                 d25 <= 0.9212, d5 > 0.02116, d5 <= 0.95691]
#             ),
#             (0.11512893564175614, 200000000000000010732324408786411, Quality: 0.11512893564175614,
#             [d3 > -0.09924, d3 <= 0.99352, d25 > -0.71731,
#                 d25 <= 0.89002, d5 > 0.01437, d5 <= 0.94982]
#             ), (0.1150558842866535, 200000000000000010732324408786410, Quality: 0.1150558842866535,
#             [d3 > -0.09924, d3 <= 0.99352, d25 > -0.71731,
#                 d25 <= 0.89002, d5 > 0.14919, d5 <= 0.94982]
#             ), (0.1150558842866535, 200000000000000010732324408786303, Quality: 0.1150558842866535,
#             [d3 > 0.0, d3 <= 0.99352, d25 > -0.70352,
#                 d25 <= 0.9212, d5 > 0.19672, d5 <= 0.95691]
#             ), (0.11330265176419019, 200000000000000010732324408786412, Quality: 0.11330265176419019,
#             [d3 > -0.09924, d3 <= 0.99352, d25 > -0.71731,
#                 d25 <= 0.89002, d5 > 0.0, d5 <= 0.94982]
#             ), (0.11220688143765065, 200000000000000010732324408786528, Quality: 0.11220688143765065,
#             [d3 > -0.50694, d3 <= 0.99352, d25 > -0.71731,
#                 d25 <= 0.87579, d5 > 0.12993, d5 <= 0.94705]
#             ), (0.10709328658046605, 200000000000000010732324408786465, Quality: 0.10709328658046605,
#             [d3 > 0.0, d3 <= 0.99352, d25 > -0.33942,
#                 d25 <= 0.9212, d5 > 0.03759, d5 <= 0.93598]
#             ), (0.10658192709474756, 200000000000000010732324408786529, Quality: 0.10658192709474756,
#             [d3 > -0.50694, d3 <= 0.99352, d25 > -0.71731,
#                 d25 <= 0.87579, d5 > 0.0, d5 <= 0.94705]
#             ), (0.10650887573964499, 200000000000000010732324408786337, Quality: 0.10650887573964499,
#             [d3 > -0.78824, d3 <= 0.99352, d25 > -0.71731,
#                 d25 <= 0.86923, d5 > 0.10526, d5 <= 0.94616]
#             ), (0.10607056760902911, 200000000000000010732324408786464, Quality: 0.10607056760902911,
#             [d3 > 0.0, d3 <= 0.99352, d25 > -0.33942,
#                 d25 <= 0.9212, d5 > 0.05319, d5 <= 0.93598]
#             ), (0.10475564321718162, 200000000000000010732324408786530, Quality: 0.10475564321718162,
#             [d3 > -0.50694, d3 <= 0.99352, d25 > -0.71731,
#                 d25 <= 0.87579, d5 > 0.0, d5 <= 0.94705]
#             ), (0.10322156476002625, 200000000000000010732324408786280, Quality: 0.10322156476002625,
#             [d3 > 0.0, d3 <= 0.99352, d25 > -0.70352,
#                 d25 <= 0.9212, d5 > 0.03759, d5 <= 0.88237]
#             ), (0.10190664036817879, 200000000000000010732324408786409, Quality: 0.10190664036817879,
#             [d3 > -0.09924, d3 <= 0.99352, d25 > -0.71731,
#                 d25 <= 0.89002, d5 > 0.37201, d5 <= 0.94982]
#             ), (0.10139528088246033, 200000000000000010732324408786279, Quality: 0.10139528088246033,
#             [d3 > 0.0, d3 <= 0.99352, d25 > -0.70352,
#                 d25 <= 0.9212, d5 > 0.02116, d5 <= 0.88237]
#             ), (0.09905763751917593, 200000000000000010732324408786527, Quality: 0.09905763751917593,
#             [d3 > -0.50694, d3 <= 0.99352, d25 > -0.71731,
#                 d25 <= 0.87579, d5 > 0.34962, d5 <= 0.94705]
#             ), (0.09905763751917593, 200000000000000010732324408786336, Quality: 0.09905763751917593,
#             [d3 > -0.78824, d3 <= 0.99352, d25 > -0.71731,
#                 d25 <= 0.86923, d5 > 0.34109, d5 <= 0.94616]
#             ), (0.09905763751917593, 200000000000000010732324408786302, Quality: 0.09905763751917593,
#             [d3 > 0.0, d3 <= 0.99352, d25 > -0.70352,
#                 d25 <= 0.9212, d5 > 0.39613, d5 <= 0.95691]
#             ), (0.09803491854773905, 200000000000000010732324408786338, Quality: 0.09803491854773905,
#             [d3 > -0.78824, d3 <= 0.99352, d25 > -0.71731,
#                 d25 <= 0.86923, d5 > 0.0, d5 <= 0.94616]
#             ), (0.09752355906202059, 200000000000000010732324408786438, Quality: 0.09752355906202059,
#             [d3 > -0.09924, d3 <= 0.99352, d25 > -0.34146,
#                 d25 <= 0.89002, d5 > 0.01997, d5 <= 0.92442]
#             )
#         ]
#             ),
#             (['d3', 'd25', 'd2'],
#             [(0.06406603842501274, 200000000000000010732324408785872, Quality: 0.06406603842501274,
#             [d3 > 0.0, d3 <= 0.99352, d25 > 0.20645, d25 <= 0.9212, d2 > -0.05, d2 <= 0.36724]), (0.06121703557600991, 200000000000000010732324408785881, Quality: 0.06121703557600991, [d3 > -0.09924, d3 <= 0.99352, d25 > 0.17308, d25 <= 0.89002, d2 > -0.06343, d2 <= 0.29167]), (0.05836803272700705, 200000000000000010732324408785991, Quality: 0.05836803272700705, [d3 > -0.78824, d3 <= 0.99352, d25 > 0.16813, d25 <= 0.86923, d2 > -0.07572, d2 <= 0.36924]), (0.05836803272700705, 200000000000000010732324408785968, Quality: 0.05836803272700705, [d3 > -0.50694, d3 <= 0.99352, d25 > 0.17041, d25 <= 0.87579, d2 > -0.07572, d2 <= 0.35232]), (0.052670027029001365, 200000000000000010732324408785927, Quality: 0.052670027029001365, [d3 > 0.0, d3 <= 0.99352, d25 > -0.00621, d25 <= 0.9212, d2 > -0.07896, d2 <= 0.11042]), (0.05245087296369347, 200000000000000010732324408785873, Quality: 0.05245087296369347, [d3 > 0.0, d3 <= 0.99352, d25 > 0.20645, d25 <= 0.9212, d2 > -0.63636, d2 <= 0.05812]), (0.04982102417999851, 200000000000000010732324408785913, Quality: 0.04982102417999851, [d3 > 0.0, d3 <= 0.99352, d25 > -0.33942, d25 <= 0.9212, d2 > -0.08459, d2 <= 0.09709]), (0.04982102417999851, 200000000000000010732324408785854, Quality: 0.04982102417999851, [d3 > 0.0, d3 <= 0.99352, d25 > -0.70352, d25 <= 0.9212, d2 > -0.09524, d2 <= 0.09417]), (0.04697202133099565, 200000000000000010732324408785977, Quality: 0.04697202133099565, [d3 > -0.09924, d3 <= 0.99352, d25 > -0.0153, d25 <= 0.89002, d2 > -0.08152, d2 <= 0.11042]), (0.04675286726568778, 200000000000000010732324408785882, Quality: 0.04675286726568778, [d3 > -0.09924, d3 <= 0.99352, d25 > 0.17308, d25 <= 0.89002, d2 > -0.63636, d2 <= 0.05375]), (0.045437942873840315, 200000000000000010732324408785880, Quality: 0.045437942873840315, [d3 > -0.09924, d3 <= 0.99352, d25 > 0.17308, d25 <= 0.89002, d2 > 0.0, d2 <= 0.29167]), (0.045437942873840315, 200000000000000010732324408785871, Quality: 0.045437942873840315, [d3 > 0.0, d3 <= 0.99352, d25 > 0.20645, d25 <= 0.9212, d2 > 0.0, d2 <= 0.36724]), (0.04499963474322448, 200000000000000010732324408786002, Quality: 0.04499963474322448, [d3 > 0.58182, d3 <= 0.99352, d25 > -0.11925, d25 <= 0.81147, d2 > -0.0339, d2 <= 0.35232]), (0.04499963474322448, 200000000000000010732324408785997, Quality: 0.04499963474322448, [d3 > 0.33333, d3 <= 0.89241, d25 > -0.00046, d25 <= 0.83008, d2 > -0.02811, d2 <= 0.38138]), (0.04499963474322448, 200000000000000010732324408785951, Quality: 0.04499963474322448, [d3 > 0.69868, d3 <= 0.99352, d25 > -0.63237, d25 <= 0.95243, d2 > -0.0239, d2 <= 0.1085]), (0.04499963474322448, 200000000000000010732324408785887, Quality: 0.04499963474322448, [d3 > 0.58182, d3 <= 0.99352, d25 > -0.63237, d25 <= 0.81147, d2 > -0.03397, d2 <= 0.16801]), (0.04412301848199282, 200000000000000010732324408785959, Quality: 0.04412301848199282, [d3 > -0.50694, d3 <= 0.99352, d25 > -0.0153, d25 <= 0.87579, d2 > -0.08459, d2 <= 0.11385]), (0.04258894002483746, 200000000000000010732324408785990, Quality: 0.04258894002483746, [d3 > -0.78824, d3 <= 0.99352, d25 > 0.16813, d25 <= 0.86923, d2 > 0.0084, d2 <= 0.36924]), (0.04258894002483746, 200000000000000010732324408785967, Quality: 0.04258894002483746, [d3 > -0.50694, d3 <= 0.99352, d25 > 0.17041, d25 <= 0.87579, d2 > 0.00075, d2 <= 0.35232]), (0.04215063189422162, 200000000000000010732324408785867, Quality: 0.04215063189422162, [d3 > 0.0, d3 <= 0.99352, d25 > 0.20645, d25 <= 0.9212, d2 > -0.05, d2 <= 0.05812])]), (['d25', 'd3', 'd14'], [(0.11461757615603768, 200000000000000010732324408785244, Quality: 0.11461757615603768, [d25 > -0.83314, d25 <= 0.9882, d3 > 0.40392, d3 <= 0.99352, d14 > -0.19497, d14 <= 0.775]), (0.11461757615603768, 200000000000000010732324408785193, Quality: 0.11461757615603768, [d25 > -0.46579, d25 <= 0.9882, d3 > 0.41245, d3 <= 1.0, d14 > -0.24164, d14 <= 0.54965]), (0.11176857330703482, 200000000000000010732324408785258, Quality: 0.11176857330703482, [d25 > -0.06024, d25 <= 0.9882, d3 > 0.42784, d3 <= 1.0, d14 > -0.27457, d14 <= 0.46086]), (0.1003725619110234,
