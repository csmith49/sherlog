# Smokers

# probabilistic rules

# p_stress :: stress(P) <- person(P).
!parameter p_stress : unit.
stress_int(P ; bernoulli[p_stress]) <- person(P).
stress(P) <- stress_int(P, 1.0).

# p_inf :: influences(P, P') <- friend(P, P').
!parameter p_inf : unit.
influences_int(P, P' ; bernoulli[p_inf]) <- friend(P, P').
influences(P, P') <- influences_int(P, P', 1.0).

# p_spont :: cancer_spont(P) <- person(P).
!parameter p_spont : unit.
cancer_spont_int(P ; bernoulli[p_spont]) <- person(P).
cancer_spont(P) <- cancer_spont_int(P, 1.0).

# p_smoke :: cancer_smoke(P) <- person(P).
!parameter p_smoke : unit.
cancer_smoke_int(P ; bernoulli[p_smoke]) <- person(P).
cancer_smoke(P) <- cancer_smoke_int(P, 1.0).

# classical rules

smokes(P) <- stress(P).
smokes(P) <- smokes(P'), influences(P, P').
cancer(P) <- cancer_spont(P).
cancer(P) <- smokes(P), cancer_smoke(P).