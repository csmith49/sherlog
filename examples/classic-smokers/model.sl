!parameter stress : unit.
!parameter spontaneous : unit.
!parameter comorbid : unit.
!parameter influence : unit.

# probabilistic rules
stress :: stress(X) <- person(X).
spontaneous :: asthma_spontaneous(X) <- person(X).
comorbid :: asthma_comorbid(X) <- smokes(X).
influence :: influence(X, Y) <- friend(X, Y).

# logical rules
smokes(X) <- stress(X).
smokes(X) <- influence(X, Y), smokes(Y).
asthma(X) <- asthma_spontaneous(X).
asthma(X) <- asthma_comorbid(X).

# ontological rules
!dependency smokes(X) | not_smokes(X) <- person(X).
!constraint smokes(X), not_smokes(X).
!dependency asthma(X) | not_asthma(X) <- person(X).
!constraint asthma(X), not_asthma(X).

person(p_0).
person(p_1).
person(p_2).
friend(p_0, p_1).
friend(p_1, p_2).
friend(p_2, p_0).
!evidence smokes(p_1), smokes(p_2), not_smokes(p_0), asthma(p_1), not_asthma(p_0), not_asthma(p_2).
