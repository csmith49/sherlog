# Explanation

This module is `sherlog.explanation`.

## Organization

`explanation.py` describes explanations as a combination of a pipeline (see the module `sherlog.pipe`), an observation, and a generation history.

`history.py` of an explanation is the sequence of choices made for sampling it from the posterior.

`observation.py` introduces observations, which constrain the generative process.

`semantics` is a sub-module that contains various concrete pipes for evaluation of explanations.