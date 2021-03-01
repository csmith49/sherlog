# parameters
!parameter stress : unit.
!parameter spontaneous : unit.
!parameter comorbid : unit.
!parameter influence : unit.

# probabilistic rules
stress :: stress(X) <- person(X).
spontaneous :: asthma_latent(X) <- person(X).
comorbid :: asthma_smokes(X) <- person(X).
influence :: influence(X, Y) <- friend(X, Y).

# logical rules
smokes(X) <- stress(X).
smokes(X) <- influences(X, Y), smokes(Y).
asthma(X) <- asthma_latent(X).
asthma(X) <- smokes(X), asthma_smokes(X).

# ontology
!dependency smokes(X) | not_smokes(X) <- person(X).
!dependency asthma(X) | not_asthma(X) <- person(X).
!constraint smokes(X), not_smokes(X).
!constraint asthma(X), not_asthma(X).

person(p_0).
person(p_1).
person(p_2).
person(p_3).
person(p_4).
person(p_5).
person(p_6).
person(p_7).
person(p_8).
person(p_9).
friend(p_0, p_1).
friend(p_1, p_2).
friend(p_1, p_1).
friend(p_1, p_3).
friend(p_2, p_0).
friend(p_2, p_3).
friend(p_4, p_1).
friend(p_5, p_3).
friend(p_6, p_1).
friend(p_7, p_2).
friend(p_8, p_1).
friend(p_9, p_1).

!evidence smokes(p_2), smokes(p_3), smokes(p_9), not_smokes(p_0), not_smokes(p_1), not_smokes(p_4), not_smokes(p_5), not_smokes(p_6), not_smokes(p_7), not_smokes(p_8), asthma(p_4), asthma(p_9), not_asthma(p_0), not_asthma(p_1), not_asthma(p_2), not_asthma(p_3), not_asthma(p_5), not_asthma(p_6), not_asthma(p_7), not_asthma(p_8).
