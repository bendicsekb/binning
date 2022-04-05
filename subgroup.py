import string
from condition import Condition
from dataset import DataSet
from pandas import DataFrame, Index

from description import Description
from search_constraints import SearchConstraints

class SubGroup(DataSet):
    def __init__(self, data: DataSet, description: list[Condition]) -> None:
        # DataSet parameters
        self.data = data
        self.index = data.targets.index
        # Own parameters
        self.description: list[Condition] = description
        self.quality: float = 0
        self.discretization_boundaries = {}
        self.depth = -1
    @property
    def descriptors(self):
        return self.data.descriptors.loc[self.index]
    @property
    def targets(self):
        return self.data.targets.loc[self.index]

    def subset(self, subset_index: Index):
        self.index = subset_index

    def add_conditions(self, index: Index, left_bound: Condition, right_bound: Condition):
        self.__add_condition(left_bound)
        self.__add_condition(right_bound)
        self.subset(index)

    def add_boundaries(self, attribute_name: string, discretization_boundaries: list[tuple], depth):
        if attribute_name not in self.discretization_boundaries:
            self.discretization_boundaries[attribute_name] = [(depth, discretization_boundaries)]
        else:
            self.discretization_boundaries[attribute_name].append((depth, discretization_boundaries))

    def __add_condition(self, new_condition: Condition):
        if (self.__in_description(new_condition)):
            for idx, condition in enumerate(self.description):
                # TODO write as a dataclass => if(condition == new_condition
                if (condition.attribute == new_condition.attribute and condition.negated == new_condition.negated and condition.value != new_condition.value):
                    self.description[idx]=new_condition
                    break
        else:
            self.description.append(new_condition)

    def __in_description(self, new_condition: str) -> bool:
        is_in: bool = False
        for condition in self.description:
            if (condition.attribute == new_condition.attribute and condition.negated == new_condition.negated):
                is_in = True
                break
        return is_in

    def __repr__(self) -> str:
        return f'Quality: {self.quality}, {self.description}'