# latent health generation per-person
health(X; dirichlet[dimension]) <- person(X).
observed_health(X; categorical[H]) <- health(X, H).

# health influences many factors (but we don't know how)
influence(X, Y; influence_nn[H1, H2]) <- health(X, H1), health(Y, H2), frined(X, Y).
asthma_risk(X; risk_nn[H]) <- health(X, H).

# probabilistic rules
stress :: smokes(X) <- person(X).
M :: smokes(X) <- influence(Y, X, M), friend(Y, X).
comorbid :: asthma(X) <- smokes(X).
R :: asthma(X) <- asthma_risk(X, R).

# ontological rules
!dependency smokes(X) | not_smokes(X) <- person(X).
!dependency asthma(X) | not_asthma(X) <- person(X).
!constraint smokes(X), not_smokes(X).
!constraint asthma(X), not_asthma(X).

person(p_0).
person(p_1).
person(p_2).
person(p_3).
friend(p_0, p_1).
friend(p_1, p_2).
friend(p_1, p_3).
friend(p_2, p_0).
friend(p_2, p_0).
!evidence not_smokes(p_0), not_smokes(p_1), not_smokes(p_2), not_smokes(p_3), asthma(p_0), asthma(p_3), not_asthma(p_1), not_asthma(p_2), observed_health(p_0, 1), observed_health(p_1, 0), observed_health(p_2, 1), observed_health(p_3, 1).