# Classic LP encoding a transitive `ancestry` relation.

# Facts
parent(xerces, brooke).
parent(brooke, damocles).

# Rules
ancestor(X, Y) <- parent(X, Y).
ancestor(X, Y) <- parent(X, Z), ancestor(Z, Y).

# Evidence
!evidence ancestor(xerces, X).