face(C; {tails, heads} <- coin_nn[C]).
color(RGB; {red, green, blue} <- color_nn[RGB]).

# parameters
!parameter urn_one_weights : unit[2].
!parameter urn_two_weights : unit[3].

draw(urn_one; {red, blue} <~ categorical[urn_one_weights]).
draw(urn_two; {red, green, blue} <~ categorical[urn_two_weights]).

outcome(heads, red, red, win).
outcome(heads, red, blue, win).
outcome(heads, red, green, win).
outcome(heads, blue, red, win).
outcome(heads, blue, blue, win).
outcome(heads, blue, green, loss).
outcome(tails, red, red, win).
outcome(tails, red, blue, loss).
outcome(tails, red, green, loss).
outcome(tails, blue, red, loss).
outcome(tails, blue, blue, win).
outocme(tails, blue, green, loss).

game(C, U1, U2, R) <-
    face(C, F),
    draw(urn_one, C1), color(U1, C1),
    draw(urn_two, C2), color(U2, C2),
    outcome(F, C1, C2, R).

!evidence game(coin, urn1, urn2, win).