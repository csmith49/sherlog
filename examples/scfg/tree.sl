# simple scfg with unknown expansion probabilities
# S -> T | A S | B B S

# dist[k] = all positive k-vecs that sum to 1
!parameter s_expansions : dist[3].

# symbols
nonterminal(s).
terminal(t).
terminal(a).
terminal(b).

# matching the expansion probabilities with the nonterms
nonterminal_expansions(s, s_expansions).

expansion_labels(s, 1, [t]).
expansion_labels(s, 2, [a, s]).
expansion_labels(s, 3, [b, b, s]).

# getting the Ith index of a list
at(1, X :: _, X).
at(I, _ :: XS, X) <- at (I - 1, XS, X).

# BUILDING THE TREE -------------------------------------------------------------------
# each node is labeled by a path from the root (in reverse order)
# -------------------------------------------------------------------------------------

# choose the expansion of a nonterminal at node N
expand(S, N; categorical[E]) <- nonterminal_expansions(S, E), node(N), label(N, S).

# expand
node(I :: N), label(I :: N, S) <- expand(S', N, E), expansion_labels(S', E, L), at(I, L, S).

# root the tree
node([0]).
label([0], s).

# EXPLORING THE TREE ------------------------------------------------------------------
# node labels give natural definitions for parent and sibling relation
# -------------------------------------------------------------------------------------

parent(N, _ :: N).
sibling(I :: N, (I + 1) :: N).