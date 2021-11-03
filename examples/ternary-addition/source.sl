!parameter left_weights : unit[3].
!parameter right_weights : unit[3].

weights(left, left_weights).
weights(right, right_weights).

trit(X; {-1, 0, 1} <~ categorical[W]) <- weights(X, W).

addition(X, Y; add[X, Y]).

observe(T) <- trit(left, L), trit(right, R), addition(L, R, T).

!evidence observe(2).