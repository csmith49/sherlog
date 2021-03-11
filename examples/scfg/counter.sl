# simple scfg with unknown expansion probabilities
# S -> T | A S | B B S

# dist[k] = all positive k-vecs that sum to 1
!parameter expansions : dist[3].

# sample from categorical to choose expansion
s_to(N; categorical(expansions) @ N).

# S -> T
s(cons(t, nil), s_count(N, NN)) <- s_to(N, 1), NN = N + 1.

# S -> A S
s(cons(a, L), s_count(N, NN)) <- s_to(N, 2), NI = N + 1, s(L, s_count(NI, NN)).

# S -> B B S
s(cons(b, cons(b, L)), s_count(N, NN)) <- s_to(N, 3), NI = N + 1, s(L, s_count(NI, NN)).

# defining the output
word(W) <- s(W, s_count(0, _)).