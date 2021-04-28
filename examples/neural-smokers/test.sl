# latent health generation per-person
latent(X; gaussian[latent_mean, latent_covar]) <- person(X).
health(X; health_nn[Z]) <- latent(X, Z).
obs_health(X; normal[H, obs_sdev]) <- health(X, H).

# health influences many factors
influence(X, Y; inf_magnitude[H1, H2]) <- health(X, H1), health(Y, H2).
asthma_risk(X; risk_nn[H]) <- health(X, H).

# probabilistic rules
R :: asthma(X) <- asthma_risk(X, R).

# ontological rules
!dependency smokes(X) | not_smokes(X) <- person(X).
!dependency asthma(X) | not_smokes(X) <- person(X).
!constraint smokes(X), not_smokes(X).
!constraint asthma(X), not_asthma(X).

person(alice).
!evidence asthma(alice).