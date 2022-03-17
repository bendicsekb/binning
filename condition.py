from cmath import nan
from types import NoneType

# TODO dataclass
# <attribute> <operator> <value>
class Condition:
    def __init__(self, attribute:str=None, value:float = nan, negated:bool = False):
        self.empty = (value == nan) or type(attribute)==NoneType
        self.attribute: str = attribute
        self.value: float = value
        self.negated: bool = negated

        
    # def stronger_than(self, other) -> bool:
    #     stronger = False
    #     match [self.empty, other.empty]:
    #         case [True, False]:
    #             stronger = False
    #         case [True, True]:
    #             stronger = True
    #         case [False, True]:
    #             stronger = True
    #         case [False, False]:
    #             match [self.negated, other.negated]:
    #                 case [False, False]:
    #                     stronger = self.value > other.value
    #                 case [True, True]: 
    #                     stronger = self.value < other.value
    #                 case _:
    #                     stronger = False
    #         case _:
    #             stronger = False
    #     return stronger

    def __repr__(self) -> str:
        representation = "Empty condition"
        if (not self.empty):
            representation = f"{self.attribute} {'>' if not self.negated else '<='} {self.value}"
        return representation