# GRAMMAR PRODUCTIONS w/ 1-HOT ENCODINGS
# tree -> branch tree tree
# tree -> red
# tree -> blue

# defining the tree
leaf(red).
leaf(blue).

# also use the following relations to encode a tree:
# root(node)
# left(node, child) <- production(node, branch), left_edge(node, child)
# right(node, child) <- production(node, branch), right_edge(node, child)
# production(node, {branch, red, blue})

one_hot(branch, branch_one_hot).
one_hot(red, red_one_hot).
one_hot(blue, blue_one_hot).

productions(N, red :: []) <- production(N, red).
productions(N, blue :: []) <- production(N, blue).
productions(N, branch :: P) <- left(N, L), right(N, R), productions(L, L'), productions(R, R'), concatenate(L', R', P).

vectors(N, V) <- productions(N, P), convert_productions(P, V).
convert_productions(P :: P', V :: V') <- convert_productions(P', V'), one_hot(P, V).
convert_productions([], []).

!namespace gvae.
!parameter x : positive.



# a head pred
head(X, X :: _).

# indexing crudely
snd(X, _ :: X :: _).

# defining ground
list(1 :: (2 :: (3 :: (4 :: [])))).

# query
list(X), head(1, X)?

# evidence
!evidence list(X), head(1, X).