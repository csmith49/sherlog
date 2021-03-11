!parameter p : unit.

flip(coin; bernoulli[p]).

result(heads) <- flip(coin, 1.0).
result(tails) <- flip(coin, 0.0).

!evidence result(heads).
!evidence result(heads).
!evidence result(tails).