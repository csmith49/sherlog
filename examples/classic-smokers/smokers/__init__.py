from . import problog, sherlog
from .graph import Parameterization, Graph
from random import randrange
from typing import Optional
from sherlog.tooling.instrumentation import Timer
from statistics import mean
from itertools import product

default_parameterization = Parameterization(
    stress=0.2,
    spontaneous=0.1,
    comorbid=0.3,
    influence=0.3
)

def dict_product(**kwargs):
    keys, values = kwargs.keys(), kwargs.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))

def sample(size : Optional[int] = None):
    if size is not None:
        return Graph(size, default_parameterization)
    else:
        # sample a random size - close social circles usually b/t 5 and 15 people
        size = randrange(5, 15)
        return Graph(size, default_parameterization)

class Interface:
    def __init__(self, fit, log_likelihood, classification_task):
        """Interface for training and evaluating generative models.

        Parameters
        ----------
        fit : List[Graph] -> Parameterization
        log_likelihood : Parameterization -> Graph -> float
        classification_task : Parameterization -> Graph -> float
        """
        self._fit = fit
        self._log_likelihood = log_likelihood
        self._classification_task = classification_task
        self._parameterization = None

    def fit(self, samples : int, graph_size : Optional[int] = None, fit_kwargs = {}):
        training_set = [sample(size=graph_size) for _ in range(samples)]

        # save the results then return the training time
        timer = Timer()
        
        with timer:
            self._parameterization = self._fit(*training_set, **fit_kwargs)

        return timer.elapsed

    def average_log_likelihood(self, samples : int, graph_size : Optional[int] = None, ll_kwargs = {}):
        test_set = [sample(size=graph_size) for _ in range(samples)]

        results, timer = [], Timer()

        for graph in test_set:
            with timer:
                log_likelihood = self._log_likelihood(self._parameterization, graph, **ll_kwargs)
            results.append( (log_likelihood, timer.elapsed) )
        
        lls, times = zip(*results)
        return mean(lls), mean(times)

    def classification_accuracy(self, samples : int, graph_size : Optional[int] = None, class_kwargs = {}):
        test_set = [sample(size=graph_size) for _ in range(samples)]

        results, timer = [], Timer()

        for graph in test_set:
            with timer:
                (confidence, gt) = self._classification_task(self._parameterization, graph, **class_kwargs)
            score = 1.0 if abs(confidence - gt) < 0.5 else 0.0
            results.append( (score, timer.elapsed) )

        scores, times = zip(*results)
        return mean(scores), mean(times)

# capturing the interfaces
problog_interface = Interface(problog.fit, problog.log_likelihood, problog.classify_asthma)
sherlog_interface = Interface(sherlog.fit, sherlog.log_likelihood, sherlog.classify_asthma)