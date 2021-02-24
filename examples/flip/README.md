# Flip

A simple program that flips a coin with beta-distributed weight. The goal is to maximize the likelihood of the observed outcomes by tuning the parameters of the beta distribution.

## Structure

By default, this directory consists of 4 experiment files:

1. `flip.sl` encodes the Sherlog program and provides evidence,
2. `train.py` optimizes the parameters given in `flip.sl`,
3. `visualize.py` renders graphical summaries of the training results, and
4. `Makefile` ties all the above together.

## Usage

You can see the command-line options for the Python scripts by running `python3 train.py --help` and `python3 visualize.py --help`.

Run `make` to generate a training log (stored in `flip-results.jsonl`) and visualization (stored in `flip-results.html`).