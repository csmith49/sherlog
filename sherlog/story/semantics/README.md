# Semantics

This folder contains all relevant functor implementations used by Sherlog. See also `../../engine/functor.py`.

## Tensor

The default semantics that evaluates computation graphs in native [PyTorch](https://pytorch.org).

## DiCE

A straight-forward implementation of the [DiCE](https://arxiv.org/pdf/1802.05098.pdf) surrogate objective.

## Graph

Enables generation of [NetworkX](https://networkx.org) graphs representing the computation graph.

## Miser

Our extension of DiCE that folds in importance sampling and forcing.

## Types

Basic static analysis functor that provides the domains of each node in the computation graph.