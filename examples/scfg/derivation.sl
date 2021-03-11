# simple scfg with unknown expansion probabilities
# S -> T | A S | B B S

# dist[k] = all positive k-vecs that sum to 1
!parameter expansions : dist[3].

# the starting state
# first index - terminals derived so far (in reverse order)
# second index - non-terminals still to expand
state(nil, cons(s, nil)).

# sample from categorical to choose expansion
state_internal(W, S; categorical[expansions]) <- state(W, cons(s, S)).

# S -> T
state(cons(t, W), S) <- state_internal(W, S, 1).

# S -> A S
state(cons(a, W), cons(s, S)) <- state_internal(W, S, 2).

# S -> B B S
state(cons(b, cons(b, W)), cons(s, S)) <- state_internal(W, S, 3).

# L is the reversed version of R
reverse(L, R) <- rev_int(L, nil, R).
rev_int(nil, L, L).
rev_int(cons(H, T), A, R) <- rev_int(T, cons(H, A), R).

# defining the actual output
word(W) <- reverse(W, T), state(T, nil).