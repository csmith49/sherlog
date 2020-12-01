# a head pred
head(X, X :: _).

# indexing crudely
snd(X, _ :: X :: _).

# defining ground
list(1 :: (2 :: (3 :: (4 :: [])))).

# queries
snd(X, 1 :: 2 :: [])?