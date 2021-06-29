"""Contains infrastructure for evaluating generative models."""

from .model import Model, DirectModel, SingletonModel, EmbeddingModel
from .utility import minibatch
from ..instrumentation import seed, Timer
from . import datasource
from .datasource import DataSource
from ...logs import get

from statistics import mean
from typing import Optional

# construct the module logger
logger = get("tooling.evaluation")

# evaluation harnesses
def evaluate(model : Model, datasource : DataSource, train_size : Optional[int] = None, test_size : Optional[int] = None, fit_kwargs={}, log_prob_kwargs={}, run_identifier : Optional[str] = None):
    """Evaluate a model with the provided dataset.

    Parameters
    ----------
    model : Model
    datasource : DataSource

    train_size : Optional[int]
    test_size : Optional[int]
    
    fit_kwargs : Dict[str, Any]
    log_prob_kwargs : Dict[str, Any]

    run_identifier : Optional[str]

    Returns
    -------
    Dict[str, Any]
    """

    # step 0: initialize
    timer = Timer()
    results = {
        "run_identifier" : run_identifier if run_identifier is not None else seed(),
        "fit_kwargs" : fit_kwargs,
        "log_prob_kwargs" : log_prob_kwargs,
        "train_size" : train_size,
        "test_size" : test_size
    }

    # step 1: gather the samples from the datasource
    train, test = datasource.get(train_size=train_size, test_size=test_size)

    # step 2: fit the model
    with timer:
        training_results = model.fit(train, **fit_kwargs)
    results["train_time"] = timer.elapsed
    
    # model.fit is not *required* to produce results, but often will
    if training_results is not None:
        results["train_results"] = training_results

    # step 3: test the model
    test_results = []
    for sample in test:
        with timer:
            log_prob = model.log_prob(sample, **log_prob_kwargs)
        test_results.append( (log_prob, timer.elapsed) )
    log_probs, log_prob_times = zip(*test_results)

    results["test_log_prob"] = mean(log_probs)
    results["test_log_prob_time"] = mean(log_prob_times)

    # step 4: we're done!
    return results