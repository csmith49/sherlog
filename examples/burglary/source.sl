!parameter earthquake_sensitivity : unit.
!parameter burglary_sensitivity : unit.

unit(H, C) <- house(H, C).
unit(B, C) <- business(B, C).

earthquake(C; bernoulli[0.1]) <- city(C, _).
burglary(X; bernoulli[R]) <- unit(X, C), city(C, R).

alarm(X; {off, on} <~ bernoulli[earthquake_sensitivity]) <- unit(X, C), earthquake(C, 1.0).
alarm(X; {off, on} <~ bernoulli[burglary_sensitivity]) <- unit(X, _), burglary(X, 1.0).

house(white, dc).
city(dc, 0.2).

!evidence alarm(white, on).
!evidence alarm(white, off).