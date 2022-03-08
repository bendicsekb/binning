from typing import List
from Condition import Condition


class Description:
    def __init__(self) -> None:
        self.description: List[Condition]= []



    # def __init__(self):
    #     self.conditions: dict[str:Interval] = dict()

    # def add_condition(self, condition: Condition):
    #     if condition.attribute in self.conditions:
    #         self.conditions[condition.attribute].add(condition)
    #     else:
    #         self.conditions[condition.attribute]


    #     interval: Interval = self._find_interval(condition.attribute)
    #     self.conditions.append(condition)
    #     self._trim()

    # def _trim(self):
    #     self._remove_duplicate_rules()
    #     # self._remove_impossible_intervals()

    # def _remove_duplicate_rules(self):
    #     for ix, c in enumerate(self.conditions):
    #         pass

    # def _find_interval(self, attribute):
    #     l:Condition = Condition(attribute, float('-inf'), negated=True)
    #     r:Condition = Condition(attribute, float('inf'), negated=False)
    #     for c in self.conditions:
    #         if c.attribute == attribute:
    #             if not c.negated:
    #                 l = c
    #             else:
    #                 r = c
    #     if l.value > r.value:
    #         tmp = l
    #         l = r
    #         r = tmp
    #     return Interval(attribute, l, r)

