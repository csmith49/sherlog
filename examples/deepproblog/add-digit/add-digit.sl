!namespace ad.

# define our addition relation - can't rely on built-in arithmetic
add(X, Y ; add[X, Y]).

# define our digit extraction relation
digit(I ; classify_digit[I]).

# the DPL rule
addition(X, Y, Z) <- digit(X, D_X), digit(Y, D_Y), add(D_X, D_Y, Z).

# parameterized evidence for training
!evidence(image1, image2, sum in training) addition(image1, image2, sum).