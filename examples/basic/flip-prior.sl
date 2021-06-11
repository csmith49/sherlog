# Parameters
!parameter p : positive.
!parameter q : positive.

# Facts
is_coin(quarter).

# Rules
weight(C ; beta[p, q] @ weight, C) <- is_coin(C).
flip(C ; bernoulli[P] @ flip, C) <- weight(C, P).

results(C, heads) <- flip(C, 1.0).
results(C, tails) <- flip(C, 0.0).

# Evidence
!evidence results(quarter, heads).
!evidence results(quarter, heads).
!evidence results(quarter, tails).