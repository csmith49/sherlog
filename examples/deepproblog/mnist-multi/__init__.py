from sherlog.tooling.data import MNISTDataSource
from sherlog.tooling.evaluation.datasource import DataSource
from sherlog.tooling.data.mnist import Image
import sherlog.tooling.evaluation.datasource.combinators as combinators

from dataclasses import dataclass
from typing import List

from .dpl import DPLModel
from .sherlog import model as sherlog_model

# class to hold samples from the desired distribution
@dataclass
class Sample:
    left : List[Image]
    right : List[Image]
    total : int

    @classmethod
    def of_images(cls, ls, rs):
        # TODO - this sum computation is wrong, make sure it matches the semantics
        total = sum([i.label for i in ls + rs])
        return cls(ls, rs, total)

# load the mnist data (default location ok)
mnist = MNISTDataSource()

# and construct the sample datasource
def samples(size : int) -> DataSource:
    mnist_list = combinators.Map(lambda *i: i, *([mnist] * size))
    return combinators.Map(Sample.of_images, mnist_list, mnist_list)