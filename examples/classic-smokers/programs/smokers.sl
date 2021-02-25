# parameters
!parameter stress : unit.
!parameter asthma_smokes : unit.
!parameter asthma_latent : unit.
!parameter influence : unit.

# probabilistic rules
stress :: stress(X) <- person(X).
asthma_latent :: asthma_latent(X) <- person(X).
asthma_smokes :: asthma_smokes(X) <- person(X).
influence :: influence(X, Y) <- friend(X, Y).

# logical rules
smokes(X) <- stress(X).
smokes(X) <- influences(X, Y), smokes(Y).
asthma(X) <- asthma_latent(X).
asthma(X) <- smokes(X), asthma_smokes(X).

# ontology
!dependency smokes(X) || not_smokes(X) <- person(X).
!dependency asthma(X) || not_asthma(X) <- person(X).
!constraint smokes(X), not_smokes(X).
!constraint asthma(X), not_asthma(X).