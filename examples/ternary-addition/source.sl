!parameter w : unit[3].

trit(X; {-1, 0, 1} <~ categorical[w]).

addition(X, Y; add[X, Y]).

observe(T) <- trit(left, L), trit(right, R), addition(L, R, T).

!evidence observe(2).