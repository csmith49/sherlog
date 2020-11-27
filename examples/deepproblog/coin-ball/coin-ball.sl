!namespace cb.

# THE COIN

# external network for classifying image
classify_coin_ext(C ; classify_coin[C]).
classify_coin(C, heads) <- classify_coin_ext(C, 1.0).
classify_coin(C, tails) <- classify_coin_ext(C, 0.0).

# a coin flip
!parameter w : unit.
flip(C, Bernoulli[w]).

# getting the results
coin(C, heads) <- classify_coin(C, heads), flip(C, 1.0).
coin(C, tails) <- classify_coin(C, heads), flip(C, 0.0).

# THE URNS

# classifying color from RGB using external network
classify_color_ext(C ; classify_color[C]).
# requires externally-defined one-hot encoding of each color
classify_color(C, red) <- classify_color_ext(C, red_vector_ext).
classify_color(C, green) <- classify_color_ext(C, green_vector_ext).
classify_color(C, blue) <- classify_color_ext(C, blue_vector_ext).

# the priors
!parameter u1 : delta[3].
!parameter u2 : delta[2].
color(1 ; Categorical[u1]).
color(2 ; Categorical[u2]).

# getting the results
urn(I, U, red) <- classify_color(U, red), color(I, 0.0).
urn(I, U, blue) <- classify_color(U, blue), color(I, 1.0).
urn(1, U, green) <- classify_color(U, green), color(1, 2.0).

# THE GAME

# possible outcomes (long list, we have no negation)
outcome(heads, red, red, win).
outcome(heads, red, blue, win).
outcome(heads, green, red, win).
outcome(heads, green, blue, loss).
outcome(heads, blue, red, win).
outcome(heads, blue, blue, win).
outcome(tails, red, red, win).
outcome(tails, red, blue, loss).
outcome(tails, green, red, loss).
outcome(tails, green, blue, loss).
outcome(tails, blue, red, loss).
outcome(tails, blue, blue, win).

# running the game
game(C, U1, U2, R) <- coin(C, F), urn(1, U1, C1), urn(2, U2, C2), outcome(F, C1, C2, R).

# parameterized evidence for training
!evidence(coin, urn1, urn2, result in training) game(coin, urn1, urn2, result).