# Pipe

A monad-based evaluation engine supporting Sherlog.

## Data Model

Uses the following structure: values, statements, programs.

## Evaluation Model

Namespaces let us select functions or callables to apply to values.

Monads tell us how to combine values and apply functions.

Semantics combine namespaces and monads to provide a complete evaluation context.

## Serialization

Pipe uses a JSON-based serialization format. Values, Statements, and Programs contain `load` class methods, and instances have a `dump` method. Functions consume and produce objects generated via Python's default `json` library.

Put the schema here eventually.