!namespace gvae.

# GRAMMAR PRODUCTIONS w/ 1-HOT ENCODINGS
# tree -> branch tree tree
# tree -> red
# tree -> blue

# also use the following relations to encode a tree:
# root(N), left_edge(P, C), right_edge(P, C), production(N, {branch, red, blue})

# explicitly-listed one-hot encodings
one_hot(branch, branch_one_hot).
one_hot(red, red_one_hot).
one_hot(blue, blue_one_hot).

# converting a tree to a list of productions in pre-order
productions(N, red :: []) <- production(N, red).
productions(N, blue :: []) <- production(N, blue).
productions(N, branch :: P) <- left(N, L), right(N, R), productions(L, L'), productions(R, R'), concatenate(L', R', P).

# convert production list to list of vectors
vectors(N, V) <- productions(N, P), convert_productions(P, V).
convert_productions(P :: P', V :: V') <- convert_productions(P', V'), one_hot(P, V).
convert_productions([], []).

# encode
encode_mean(N ; ext_encode_m[V]) <- vectors(N, V).
encode_sdev(N ; ext_encode_s[V]) <- vectors(N, V).

# sample from latent space
latent(N ; Normal[M, S]) <- encode_mean(N, M), encode_sdev(N, S).

# decode to stack
decode(N ; ext_decode[Z]) <- latent(N, Z).

# process stack

# external function for picking the production from a vector
get_production(V, ext_prod[V]).



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