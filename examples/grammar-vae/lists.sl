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