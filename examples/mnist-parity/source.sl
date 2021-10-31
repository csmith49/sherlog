digit(I; {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} <- digit_nn[I]).

even(0).
even(2).
even(4).
even(6).
even(8).

odd(1).
odd(3).
odd(5).
odd(7).
odd(9).

observe(I, even) <- digit(I, D), even(D).
observe(I, odd) <- digit(I, D), odd(D).

!evidence observe(image, even).
!evidence observe(image, odd).