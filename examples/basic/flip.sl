# Parameters
!parameter p : unit.

# Rules
flip(coin; bernoulli[p]).

result(heads) <- flip(coin, 1.0).
result(tails) <- flip(coin, 0.0).

# Evidence
!evidence result(heads).
!evidence result(heads).
!evidence result(tails).