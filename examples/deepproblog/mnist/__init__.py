from .nn import MNISTNetwork

from sherlog.tooling.data import MNISTDataSource
from sherlog.tooling.data.mnist import Image
import sherlog.tooling.evaluation.datasource.combinators as combinators

from dataclasses import dataclass

from dpl import DPLModel

# class to hold samples from the desired distribution
@dataclass
class Sample:
    left : Image
    right : Image
    total : int

    @classmethod
    def of_images(cls, left, right):
        total = left.label + right.label
        return cls(left, right, total)

# load the mnist data (default location ok)
mnist = MNISTDataSource()

# and construct the sample datasource
samples = combinators.Map(Sample.of_images, mnist, mnist)