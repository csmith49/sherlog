# Scripts

This module is `sherlog.scripts`. Operations described here are intended for use on the command line, not for tools importing `sherlog` as a module.

## Organization

The entry point is `cli.py`, which constructs a Click group that parameterizes some basic set-up.

`train.py` gives a simple training script for quick testing.

`explain.py` supports sampling explanations from the posterior.

`analyze.py` produces graphs for instrumented Sherlog programs.