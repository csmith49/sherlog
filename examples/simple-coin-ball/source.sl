# parameters
!parameter weight : unit.
!parameter urn_one_weights : unit[2].
!parameter urn_two_weights : unit[3].

flip(coin; {tails, heads} <~ bernoulli[weight]).
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

game(R) <-
    flip(coin, F),
    draw(urn_one, B1),
    draw(urn_two, B2),
    outcome(F, B1, B2, R).

!evidence game(win).
!evidence game(loss).