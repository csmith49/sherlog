# our observations
parent(xerces, brooke).
parent(brooke, damocles).

# rules for ancestry
ancestor(X, Y) <- parent(X, Y).
ancestor(X, Y) <- parent(X, Z), ancestor(Z, Y).

# evidence
!evidence ancestor(xerces, X).