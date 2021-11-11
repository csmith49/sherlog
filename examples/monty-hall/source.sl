pick(; {0, 1, 2} <~ discrete[3]).

swap(0; {1, 2} <~ discrete[2]).
swap(1; {0, 2} <~ discrete[2]).
swap(2; {0, 1} <~ discrete[2]).

strategy(swap, D) <- pick(D'), swap(D', D).
strategy(keep, D) <- pick(D).

outcome(0, win).
outcome(1, loss).
outcome(2, loss).

play(S, R) <- strategy(S, D), outcome(D, R).

!evidence play(swap, win).
!evidence play(swap, loss).
!evidence play(keep, win).
!evidence play(keep, loss).