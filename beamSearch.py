from ast import Sub
import heapq
from typing import List
from pandas import Index
from Condition import Condition

from DataSet import DataSet
from Description import Description
from QualityMeasure import QualityMeasure
from ResultSet import ResultSet
from SearchConstraints import SearchConstraints
from SubGroup import SubGroup


def beamSearch(D: DataSet, phi: QualityMeasure,  P: SearchConstraints) -> ResultSet:
    F: List[SubGroup] = list() # Result set
    S: List[SubGroup] = list() # Candidate set
    I: Description = Description([]) # Empty description

    heapq.heappush(S, (0, SubGroup(D, I))) # S = S U I_empty 
    
    
    for _ in range(P.depth):
        beam: List[SubGroup] = []

        while (len(S) > 0):
            s: SubGroup = selectSeed(P, S)

            R: List[SubGroup] = generateRefinements(s, D, P)
            for r in R:
                score = phi.run_on(r)
                r.quality = score
                addToBeam(P, beam, r, score)
                addToResultSet(P, F, r, score)
        addToCandidateSet(S, beam)
    
    return F


def selectSeed(P: SearchConstraints, S: SubGroup) -> SubGroup:
    return S.pop(0)[1] # S <- S \ s 

def generateRefinements(s: SubGroup, D:DataSet, P:SearchConstraints) -> List[SubGroup]:
    attribute_names = D.descriptors.columns
    refinements = []
    for attribute_name in attribute_names:
        sorted = s.descriptors.loc[s.descriptors[attribute_name].argsort()]
        # generate binary splits ordered from smallest to largest (10000 11000 11100 11110)
        splits: List[Index] = P.quantizer.discretize(sorted[attribute_name])
        # generate all possible intervals from combining splits
        intervals: List[Index] = generateIntervals(sorted.index, splits)
        # for every interval make a refinement
        for interval in intervals:
            first = D.descriptors[interval].head(1)[attribute_name]
            last = D.descriptors[interval].tail(1)[attribute_name]
            # 1. make new conditions based on interval
            left_bound: Condition = Condition(attribute_name, value=first)
            right_bound: Condition = Condition(attribute_name, value=last, negated=True)
            # 2. check if conditions can be added to description, if so, add to description and subset the data
            refinement: SubGroup = s
            refinement.add_conditions(interval, left_bound, right_bound)

            refinements.append(refinement)
        

def generateIntervals(all:Index, splits: List[Index]) -> List[Index]:
    # add splits directly
    intervals: List[Index] = splits
    # add all negations of splits
    for split in splits:
        intervals.append(all.difference(split, sort=False))
    # add intersection of splits
    if (len(splits) > 1):
        for i, split in enumerate(splits[1:]):
            intervals.append(split.intersection(all.difference(splits[i], sort=False), sort=False))
    return intervals


def addToBeam(P:SearchConstraints, beam: List[SubGroup], r: SubGroup, score: float):
    if (len(beam) < P.width):
        heapq.heappush(beam, (score, r))
    else:
        heapq.heappushpop(beam, (score, r))


def addToCandidateSet(S: List[SubGroup], beam: List[SubGroup]):
    S.extend(beam)


def addToResultSet(P: SearchConstraints, F: ResultSet, r: SubGroup, score: float):
    if (len(F) < P.q):
        heapq.heappush(F, (score, r))
    else:
        heapq.heappushpop(F, (score, r))

