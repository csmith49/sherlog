# SherLog

A declarative, probabilistic, deep logic programming language.

## Introduction

We'll put a well-motivated example here.

For further examples, see the `./examples` directory.

### Architecture

SherLog is constructed as the union of three separate sub-components:

1. The externally-guarded generative logic programming core, Watson (`./lib/watson`),
2. A translation layer converting problem files to Watson programs and the resulting abducted obligations to generative models (`./lib/sherlog`),
3. And a host of inference techniques built around the produced generative models (`./sherlog`).

Sub-components `1.` and `2.` are implemented in [OCaml](https://ocamlverse.github.io) for speed. Sub-component `3.` is implemented in [Python](https://www.python.org) so as to stand on the shoulders of [PyTorch](https://pytorch.org) and [Pyro](https://pyro.ai). Communication is handled through [JSON](https://www.json.org/json-en.html) serialization sent via a local network socket.

## Installation and Building

This section assumes you have the latest versions of [OPAM](https://opam.ocaml.org) (installation instructions [here](https://opam.ocaml.org/doc/Install.html)) and [Python](https://www.python.org).

First, we build the dependencies for the OCaml core. Simply run `opam install . --deps-only`. We can then build the daemon `./sherlog` communicates with by running `make server`.

Next, we can get the Python dependencies and install the SherLog package by running `pip install .`.

## Usage

Open a Python interpreter and `import sherlog` to get started.

### Core Syntax

Base rule structure will go here.

### Inference Syntax

Specialized inference syntax will go here.