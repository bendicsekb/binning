from ast import Sub
import copy
import heapq
from typing import List
from pandas import Index
from condition import Condition

from dataset import DataSet
from description import Description
from quality_measure import QualityMeasure
from search_constraints import SearchConstraints
from subgroup import SubGroup


def beamSearch(D: DataSet, phi: QualityMeasure,  P: SearchConstraints) -> list[SubGroup]:
    F: list[SubGroup] = list() # Result set
    S: list[SubGroup] = list() # Candidate set
    I: list[Condition] = list() # Empty description
    unique_counter = int(2e16-1)
    heapq.heappush(S, (0, 0, SubGroup(D, I))) # S = S U I_empty 
    
    for depth in range(P.depth):
        beam: list[SubGroup] = []

        while (len(S) > 0):
            s: SubGroup = selectSeed(P, S)

            # R: list[SubGroup] = generateRefinements(s, D, P, depth)
            for r in generateRefinements(s, D, P, depth):
                unique_counter -= 1
                score = phi.run_on(r)
                r.quality = score
                r.depth = depth
                addToBeam(P, beam, r, score, unique_counter)
                addToResultSet(P, F, r, score, unique_counter)
        addToCandidateSet(S, beam)
    
    return F


def modifiedBeamSearch(D: DataSet, phi: QualityMeasure,  P: SearchConstraints, descriptor_sets:list) -> list[SubGroup]:
    F = list() # Result set
    I: list[Condition] = list() # Empty description
    unique_counter = int(2e32)
    for descriptor_set in descriptor_sets:
        S: list[SubGroup] = list() # Candidate set
        heapq.heappush(S, (0, 0, SubGroup(D, I)))
        for depth, descriptor in enumerate(descriptor_set):
            beam: list[SubGroup] = []
            while (len(S) > 0):
                s: SubGroup = selectSeed(P, S)
                P.use_columns = [descriptor]
                for r in generateRefinements(s, D, P, depth):
                    unique_counter -= 1
                    score = phi.run_on(r)
                    r.quality = score
                    r.depth = depth
                    addToBeam(P, beam, r, score, unique_counter)
            addToCandidateSet(S, beam)
            # [heapq.heappop(beam) for _ in range(len(beam))]
        S.sort()
        F.append((descriptor_set, S[::-1]))

    return F
def selectSeed(P: SearchConstraints, S: SubGroup) -> SubGroup:
    return S.pop(0)[2] # S <- S \ s 

def generateRefinements(s: SubGroup, D:DataSet, P:SearchConstraints, depth: int) -> list[SubGroup]:
    attribute_names = P.use_columns
    # refinements = []
    for attribute_name in attribute_names:
        sorted = s.descriptors.loc[s.index[s.descriptors[attribute_name].argsort()]]
        # generate splits with tuples in form (from:to)
        splits: list[tuple] = P.quantizer.discretize(sorted[attribute_name])
        boundaries = [(sorted[attribute_name][sorted.index[split[0]]], sorted[attribute_name][sorted.index[split[1]]]) for split in splits]
        # generate all possible intervals from combining splits
        intervals: list[Index] = P.quantizer.generate_intervals(sorted[attribute_name], splits)
        # for every interval make a refinement
        for interval in intervals:
            first = D.descriptors.loc[interval[0]][attribute_name]
            last = D.descriptors.loc[interval[-1]][attribute_name]
            # 1. make new conditions based on interval
            left_bound: Condition = Condition(attribute_name, value=first)
            right_bound: Condition = Condition(attribute_name, value=last, negated=True)
            # 2. check if conditions can be added to description, if so, add to description and subset the data
            refinement: SubGroup = copy.deepcopy(s)
            refinement.add_conditions(interval, left_bound, right_bound)
            refinement.add_boundaries(attribute_name, boundaries, depth)

            yield refinement
    # return refinements
        

# def generateIntervals(all:Index, splits: list[Index]) -> list[Index]:
#     # add splits directly
#     intervals: list[Index] = splits
#     # add all negations of splits
#     for split in splits:
#         intervals.append(all.difference(split, sort=False))
#     # add intersection of splits
#     if (len(splits) > 1):
#         for i, split in enumerate(splits[1:]):
#             intervals.append(split.intersection(all.difference(splits[i], sort=False), sort=False))
#     return intervals


def addToBeam(P:SearchConstraints, beam: list[SubGroup], r: SubGroup, score: float, unique_counter: int):
    if (len(beam) < P.width):
        heapq.heappush(beam, (score, unique_counter, r))
    else:
        heapq.heappushpop(beam, (score, unique_counter, r))


def addToCandidateSet(S: list[SubGroup], beam: list[SubGroup]):
    S.extend(beam)


def addToResultSet(P: SearchConstraints, F: list[SubGroup], r: SubGroup, score: float, unique_counter: int):
    if (len(F) < P.q):
        heapq.heappush(F, (score, unique_counter, r))
    else:
        heapq.heappushpop(F, (score, unique_counter, r))

