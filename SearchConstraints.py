from Discretization import Discretizer
from QualityMeasure import QualityMeasure


class SearchConstraints:
    def __init__(self, depth:int, width:int, q:int, minimum_coverage:float, quality_measure:QualityMeasure, quantizer: Discretizer) -> None:
        self.depth = depth
        self.width = width
        self.q = q
        self.minimum_coverage = minimum_coverage
        self.quality_measure = quality_measure
        self.quantizer = quantizer