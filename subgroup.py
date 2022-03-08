from ast import List
from condition import Condition
from dataset import DataSet
from pandas import DataFrame, Index

from description import Description
from search_constraints import SearchConstraints

class SubGroup(DataSet):
    def __init__(self, data: DataSet, description: list[Condition]) -> None:
        # DataSet parameters
        self.descriptors = data.descriptors
        self.targets = data.targets
        # Own parameters
        self.description: list[Condition] = description
        self.quality: float = 0

    def subset(self, subset_index: Index):
        self.descriptors = self.descriptors[subset_index]
        self.targets = self.targets[subset_index]

    def add_conditions(self, index: Index, left_bound: Condition, right_bound: Condition):
        self.__add_condition(left_bound)
        self.__add_condition(right_bound)
        self.subset(index)

    def __add_condition(self, new_condition: Condition):
        if (self.__in_description(new_condition.attribute)):
            for idx, condition in enumerate(self.description):
                # TODO write as a dataclass => if(description == condition
                if (condition.attribute == new_condition.attribute and condition.negated == new_condition.negated and condition.value != new_condition.value):
                    self.description[idx]=new_condition
                    break
        else:
            self.description.append(new_condition)

    def __in_description(self, attribute_name: str) -> bool:
        is_in: bool = False
        for condition in self.conditions:
            if (attribute_name == condition.attribute):
                is_in = True
                break
        return is_in