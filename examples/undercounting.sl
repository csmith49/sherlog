!parameter stress : unit.

stress :: stress(X) <- person(X).

!dependency stress(X) | not_stress(X) <- person(X).
!constraint stress(X), not_stress(X).

person(p_1).
person(p_2).
person(p_3).
person(p_4).

!evidence stress(p_1), stress(p_2), not_stress(p_3), not_stress(p_4).