# Inference

This module is `sherlog.inference`, and contains tooling and infrastructure to help organize the training, analysis, and inference of Sherlog programs.

## Why bother?

While Sherlog can be used to embed fairly arbitrary probabilistic relational data, the mechanisms for doing so are not uniform per-problem. This module was designed to facilitate a uniform optimization core accessed via a predictable interface.

## Organization

`optimizer.py` handles the actual optimization: parameter tracking, gradient propagation, clean-up, NaN-filtering, and so forth. Think of them as wrapping Torch optimizers (SGD, Adam, etc.) around a Sherlog program.

`batch.py` is a utility module that supports simple mini-batching.

`embedding.py` is the primary interface: they map data points to an objective object (see below).

`objective.py` represents an unevaluated optimization objective. Optimizer objects consume objectives to update the parameters of their associated Sherlog program.

## Notes

Goal: optimizer consumes objectives, which describe the necessary information for producing an optimization target. They are declarative - the how is left to the optimizer.

Embeddings should be the interface into objectives. Construct an embedding, use it in a loop to produce objectives that can be fed to an optimizer. Think of an embedding as a producer, an optimizer as a consumer, and an objective as the intermediate message between the two.