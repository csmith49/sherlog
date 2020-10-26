# latent variable and flip semantics
weight(C ; normal[m, s] @ weight, C) <- is_coin(C).
flip(C ; bern[P] @ flip, C) <- weight(C, P).

# minor conversion of random samples
results(C, heads) <- flip(C, true).
results(C, tails) <- flip(C, false).

# observations
is_coin(coin1).

#query
results(coin1, heads)?