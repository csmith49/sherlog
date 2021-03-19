# simple scfg with unknown expansion probabilities
# S -> T | A S | B B S

# dist[k] = all positive k-vecs that sum to 1
!parameter s_expansions : dist[3].

# symbols
nonterminal(s).
terminal(t).
terminal(a).
terminal(b).

# getting the Ith index of a list
at(1, X :: _, X).
at(I, _ :: XS, X) <- at (I - 1, XS, X).

# BUILDING THE TREE -------------------------------------------------------------------
# each node is labeled by a path from the root (in reverse order)
# -------------------------------------------------------------------------------------

# pick an rhs sequence by sampling conditioned on parent and prev sibling
# assume expansion_nn returns sequence of labels that terminate in an epsilon
# 1. P - parent
# 2. S - sibling
# 3. L - label
# 4. A - inherited attributes

expansion_choice(N; expansion_nn[P, S, L, A]) <-
    node(N), label(N, L),
    parent(P, N), sibling(S, N),
    inherited(N, A).

node(I :: N), label(I :: N, L) <- expansion_choice(N, C), at(I, C, L).

# root the tree
node([0]).
label([0], s).

# EXPLORING THE TREE ------------------------------------------------------------------

parent(N, _ :: N).
sibling(I :: N, (I + 1) :: N).
first(0 :: N).

# INHERITED ATTRIBUTES ----------------------------------------------------------------

# the first child has inherited attributes only from the parent
inherited(N; inherited_attrs[L, [], I]) <-
    node(N), label(N, L), parent(P, N), inherited(P, I),
    first(N).

# other children inherit attributes from their already-generated sibling
inherited(N; inherited_attrs[L, S, I]) <-
    node(N), label(N, L), parent(P, N), inherited(P, I),
    sibling(N', N), synthesized(N', S).

# SYNTHESIZED ATTRIBUTES --------------------------------------------------------------

# attribute synthesis depends on attributes from all children
synthesized(N, synthesized_attrs[L, S, I]) <-
    node(N), label(N, L),
    inherited(N, I),
    synthesized_from_children(N, S).

# synthesized attributes from children "starts" at 0 and goes to epsilon
# we use an aux function to limit the recursion

# case 1: no children
sfc_aux(N, [], true) <- epsilon(0 :: N).

# case 2: starting child
sfc_aux(N, (0, S) :: [], false) <-
    node(0 :: N), synthesized(0 :: N, S).

# case 3: "next" child
sfc_aux(N, (I, S) :: R, false) <-
    node(I :: N), synthesized(I :: N, S),
    R = (I - 1, _) :: _, sfc_aux(N, R, false).

# case 4: terminal child
sfc_aux(N, R, true) <-
    R = (I, _) :: _, epsilon((I + 1) :: N).

# reverse the resulting accumulated symbol sets
synthesized_from_children(N, S) <- sfc_aux(N, R, true), reverse(R, S).