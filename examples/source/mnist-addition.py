digit(X; {0, 1} <- digit_nn[X]).

addition(X, Y; add[X', Y']) <- digit(X, X'), digit(Y, Y').

!evidence addition(left, right, total).
