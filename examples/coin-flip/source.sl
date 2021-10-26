!parameter p : unit.

flip(coin; {tails, heads} <~ bernoulli[p]).
result(F) <- flip(coin, F).

!evidence result(heads).
!evidence result(tails).