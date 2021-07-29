digit_probs(X; digit_nn[X] @ digit_probs, X).
digit(X; categorical[P] @ digit, X, P) <- digit_probs(X, P).

add(X, Y; add[X, Y] @ add, X, Y).

addition(X, Y, Z) <- digit(X, X2), digit(Y, Y2), add(X2, Y2, Z).

!evidence addition(left, right, total).