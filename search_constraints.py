from discretization import Discretizer

class SearchConstraints:
    def __init__(self, depth:int, width:int, q:int, minimum_coverage:float, quality_measure, quantizer: Discretizer, use_columns=list[str]) -> None:
        self.depth = depth
        self.width = width
        self.q = q
        self.minimum_coverage = minimum_coverage
        self.quality_measure = quality_measure
        self.quantizer = quantizer
        self.use_columns = use_columns