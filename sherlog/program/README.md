# Program

This module is `sherlog.program`.

## Organization

`evidence.py` is a JSOn representation of a conjunction of atoms.

`parameter.py` is a wrapper around tensors intended to be optimized.

`posterior.py` simple posterior used for sampling explanations.

`program.py` puts it all together, combining a set of rules (stored in a JSON representation for easy serialization to `sherlog-server`), a set of parameters for optimization, and a posterior.