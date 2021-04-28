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

# ontological rules
!dependency smokes(X) | not_smokes(X) <- person(X).
!dependency asthma(X) | not_asthma(X) <- person(X).
!constraint smokes(X), not_smokes(X).
!constraint asthma(X), not_asthma(X).