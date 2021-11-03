digit(X; {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} <- digit_nn[X]).

addition(X, Y; add[X', Y']) <- digit(X, X'), digit(Y, Y').

!evidence addition(left, right, 7).
