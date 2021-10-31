digit(X; {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} <- digit_nn[X]).

observe(I, D) <- digit(I, D).

!evidence observe(image, T).
