# weight parameters
! p : positive.
! q : positive.

# latent variable and flip semantics
weight(C ; beta[p, q] @ weight, C) <- is_coin(C).
flip(C ; bernoulli[P] @ flip, C) <- weight(C, P).

# minor conversion of random samples
results(C, heads) <- flip(C, 1.0).
results(C, tails) <- flip(C, 0.0).

# observations
is_coin(coin1).

#query
results(coin1, heads)?

# evidence
! results(coin1, heads).
! results(coin1, heads).
! results(coin1, tails).