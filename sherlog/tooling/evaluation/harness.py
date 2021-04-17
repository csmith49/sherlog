from .. import logs
from .kernel import Kernel
from .datasource import DataSource
from .distribution import Distribution
from typing import TypeVar, Generic, Callable
from itertools import product

T = TypeVar('T')

logger = logs.get("evaluation.harness")

class Harness(Generic[T]):
    def __init__(self, distribution : Distribution[T], datasource : DataSource[T]):
        """
        Parameters
        ----------
        distribution : Distribution[T]
        datasource : DataSource[T]
        """
        self.distribution = distribution
        self.datasource = datasource

        # print logs and wrap to record fit time
        self.distribution.fit(self.datasource.train())
        
    def average_log_likelihood(self):
        """Compute the average log-likelihood of all test data.

        Returns
        -------
        float
        """
        lls = [self.distribution.log_likelihood(x) for x in self.datasource.test()]
        return sum(lls) / len(lls)

    def maximum_mean_discrepancy(self, kernel : Kernel[T]) -> float:
        """Computes MMD between the generative and data distributions using the provided kernel.

        Parameters
        ----------
        kernel : Kernel[T]

        Returns
        -------
        float
        """

        test = self.datasource.test()

        # sample that many from the generative distribution
        gen = [self.distribution.sample() for _ in range(len(test))]

        # pairwise comparisons
        gen_kernels = [kernel(x, y) for x, y in product(gen, gen)]
        test_kernels = [kernel(x, y) for x, y in product(test, test)]
        prod_kernels = [kernel(x, y) for x, y in product(gen, test)]

        # and expectations
        gen_exp = sum(gen_kernels) / len(gen_kernels)
        test_exp = sum(test_kernels) / len(test_kernels)
        prod_exp = sum(prod_kernels) / len(prod_kernels)

        return gen_exp + test_exp - 2 * prod_exp