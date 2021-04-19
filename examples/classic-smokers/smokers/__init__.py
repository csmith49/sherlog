from . import problog, sherlog
from .graph import Parameterization, Graph
from random import randrange
from typing import Optional
from sherlog.tooling.instrumentation import Timer
from statistics import mean

default_parameterization = Parameterization(
    stress=0.2,
    spontaneous=0.1,
    comorbid=0.3,
    influence=0.3
)

def sample(size : Optional[int] = None):
    if size is not None:
        return Graph(size, default_parameterization)
    else:
        # sample a random size - close social circles usually b/t 5 and 15 people
        size = randrange(5, 15)
        return Graph(size, default_parameterization)

class Interface:
    def __init__(self, fit, log_likelihood):
        self._fit = fit
        self._log_likelihood = log_likelihood
        self._parameterization = None

    def fit(self, samples : int, graph_size : Optional[int] = None):
        training_set = [sample(size=graph_size) for _ in range(samples)]

        # save the results then return the training time
        timer = Timer()
        
        with timer:
            self._parameterization = self._fit(*training_set)

        return timer.elapsed

    def average_log_likelihood(self, samples : int, graph_size : Optional[int] = None):
        test_set = [sample(size=graph_size) for _ in range(samples)]

        results, timer = [], Timer()

        for graph in test_set:
            with timer:
                log_likelihood = self._log_likelihood(self._parameterization, graph)
            results.append( (log_likelihood, timer.elapsed) )
        
        lls, times = zip(*results)
        return mean(lls), mean(times)

problog_interface = Interface(problog.fit, problog.log_likelihood)
sherlog_interface = Interface(sherlog.fit, sherlog.log_likelihood)