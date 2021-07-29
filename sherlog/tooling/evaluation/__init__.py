"""Contains infrastructure for evaluating generative models."""

from .model import Model, DirectModel, SingletonModel, EmbeddingModel
from .utility import minibatch
from ..instrumentation import seed, Timer
from . import datasource
from .datasource import DataSource
from ...logs import get
from torch import stack

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
    logger.info(f"Initializing evaluation of model {model} on datsource {datasource}...")
    timer = Timer()
    results = {
        "run_identifier" : run_identifier if run_identifier is not None else seed(),
        "fit_kwargs" : fit_kwargs,
        "log_prob_kwargs" : log_prob_kwargs,
        "train_size" : train_size,
        "test_size" : test_size
    }

    # step 1: gather the samples from the datasource
    logger.info(f"Generating {train_size}/{test_size} (train/test) data...")
    train, test = datasource.get(train_size=train_size, test_size=test_size)

    # step 2: fit the model
    logger.info("Fitting model...")
    with timer:
        training_results = model.fit(train, **fit_kwargs)
    results["train_time"] = timer.elapsed
    logger.info(f"Model fit to data in {results['train_time']} seconds.")
    
    # model.fit is not *required* to produce results, but often will
    if training_results is not None:
        logger.info(f"Auxiliary results: {training_results}")
        results["train_results"] = training_results

    # step 3: test the model
    logger.info("Testing model...")
    test_results = []
    for sample in test:
        with timer:
            log_prob = model.log_prob(sample, **log_prob_kwargs)
        test_results.append( (log_prob, timer.elapsed) )
        logger.info(f"Sample {sample} log-likelihood {log_prob:f} computed in {timer.elapsed} seconds.")
    logger.info("Testing done. Collating results...")
    log_probs, log_prob_times = zip(*test_results)

    results["test_log_prob"] = stack(log_probs).mean().item()
    results["test_log_prob_time"] = mean(log_prob_times)

    # step 4: we're done!
    return results