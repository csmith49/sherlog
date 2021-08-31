# Pipe

This module is `sherlog.pipe`, an evaluation engine for sequences of assignment statements.

## Why bother?

The computational artifact-of-interest in Sherlog are *stochastic computation graphs*: directed, acyclic graphs whose edges capture dependencies and whose nodes are annotated with arbitrary functions. Existing implementations of stochastic computation graphs, for various reasons, are not suited to our purpose. This module conceptualizes SCGs as sequences of assignments (their natural form when produced via Sherlog's abductive resolution), and provides utilities for their evaluation and analysis.

## Data model

Values - variables and literals

Statements - always of the form `target <- function(arguments)`.

Pipelines - just a sequence of statements

## Evaluation model

Namespace - look up functions

Pipe - not a monad!

Semantics - combine a namespace and a pipe to produce semantics for evaluating a pipeline.

## Serialization

Like the rest of Sherlog, Pipe uses a JSON-based serialization format. Each object in the data model contains `to_json : t -> json` and `of_json : json -> t option` functions which perform exactly as they suggest.

The schema for this serialization is given in `#/schemas/pipeline.schema.json`, relative to the root of this repository.