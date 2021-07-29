# Sherlog

Deep, generative logic programming.

## Introduction

Sherlog is a logic programming language with *value introductions* that supports differentiable functions and probability distributions (both discrete and continuous).

We'll put a well-motivated example here.

For further examples, see the `./examples` directory.

### Architecture

SherLog is constructed as the union of three separate sub-components:

1. The externally-guarded generative logic programming core, Watson (`./lib/watson`),
2. A translation layer converting problem files to Watson programs and the resulting abducted obligations to generative models (`./lib/sherlog`),
3. And a host of inference techniques built around the produced generative models (`./sherlog`).

Sub-components `1.` and `2.` are implemented in [OCaml](https://ocamlverse.github.io) for speed. Sub-component `3.` is implemented in [Python](https://www.python.org) so as to stand on the shoulders of [PyTorch](https://pytorch.org). Communication is handled through [JSON](https://www.json.org/json-en.html) serialization sent via a local network socket.

## Installation and Building

This section assumes you have the latest versions of [OPAM](https://opam.ocaml.org) (installation instructions [here](https://opam.ocaml.org/doc/Install.html)) and [Python](https://www.python.org).

### Option 1

Just run `make`.

### Option 2

First, we can install the OCaml core with `opam install .`. This will install the executables `sherlog-interpreter` and `sherlog-server` in the default OPAM installation path (by default, at `~/.opam/default/bin`); either run `opam init` to add the installation path to your path in your shell configuration file, or run `eval $(opam env)` to add the path temporarily.

Next, we can install the SherLog Python interface by running `python3 -m pip install .`. We recommend doing so in a virtual environment.

## Usage

Open a Python interpreter and `import sherlog` to get started. Additionally, the OPAM build process installs `sherlog-interpreter` and `sherlog-server`: run them to see their command-line arguments.